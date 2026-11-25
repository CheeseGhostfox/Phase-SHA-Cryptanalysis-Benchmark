"""Universal 32-bit ARX Bytecode Virtual Machine & Exact SHA-256 Compiler for V7C Stage 1/2.

Supports batched GPU execution in pure `torch.int64` arithmetic for opcodes:
  0: NOP
  1: XOR   reg[dst] = (reg[a] ^ operand_b) & MASK32
  2: ADD32 reg[dst] = (reg[a] + operand_b) & MASK32
  3: ROTR  reg[dst] = rotr32(reg[a], imm & 31)
  4: SHR   reg[dst] = (reg[a] >> (imm & 31)) & MASK32
  5: CH    reg[dst] = ch32(reg[a], reg[b], reg[imm])
  6: MAJ   reg[dst] = maj32(reg[a], reg[b], reg[imm])

Where `operand_b = reg[b] if b >= 0 else (imm & MASK32)` so constants (`SHA256_K`, `SHA256_IV`)
fit naturally inside `(op, dst, a, b, imm)` tuples across 32 working registers.
"""

from __future__ import annotations

from typing import Tuple

import torch

from src.sha256_ref import MASK32, SHA256_IV, SHA256_K, ch32, maj32, rotr32

OP_NOP = 0
OP_XOR = 1
OP_ADD32 = 2
OP_ROTR = 3
OP_SHR = 4
OP_CH = 5
OP_MAJ = 6
NUM_OPS = 7
NUM_REGS = 32
ARX_OP_NAMES = ("NOP", "XOR", "ADD32", "ROTR", "SHR", "CH", "MAJ")


def execute_vm_batch(
    prog: torch.Tensor,
    x_inputs: torch.Tensor,
    out_reg_start: int = 16,
    out_words: int = 8,
) -> torch.Tensor:
    """Execute bytecode `prog` on batched 16-word inputs `x_inputs` `(B, 16)` in exact `int64`.

    Args:
        prog: `(L, 5)` shared program OR `(B, L, 5)` per-sample programs.
        x_inputs: `(B, 16)` int64 uint32 input words loaded into registers `0..15`.
        out_reg_start: starting register index read out as output `Y`.
        out_words: number of consecutive registers returned as `(B, out_words)`.
    """
    if x_inputs.ndim != 2 or x_inputs.shape[1] != 16:
        raise ValueError(f"Expected x_inputs shape (B, 16), got {tuple(x_inputs.shape)}")
    dev = x_inputs.device
    b_size = int(x_inputs.shape[0])

    regs = torch.zeros((b_size, NUM_REGS), dtype=torch.int64, device=dev)
    regs[:, :16] = x_inputs.to(torch.int64) & MASK32

    if prog.ndim == 2:
        # Fast path when all batch elements share one program (e.g. SHA-256 compilation)
        prog_cpu = prog.detach().to("cpu").tolist()
        for op, dst, a, b, imm in prog_cpu:
            if op == OP_NOP:
                continue
            va = regs[:, a]
            if op == OP_XOR:
                vb = regs[:, b] if b >= 0 else (int(imm) & MASK32)
                regs[:, dst] = (va ^ vb) & MASK32
            elif op == OP_ADD32:
                vb = regs[:, b] if b >= 0 else (int(imm) & MASK32)
                regs[:, dst] = (va + vb) & MASK32
            elif op == OP_ROTR:
                regs[:, dst] = rotr32(va, int(imm) & 31)
            elif op == OP_SHR:
                regs[:, dst] = (va >> (int(imm) & 31)) & MASK32
            elif op == OP_CH:
                regs[:, dst] = ch32(va, regs[:, b], regs[:, int(imm) & 31])
            elif op == OP_MAJ:
                regs[:, dst] = maj32(va, regs[:, b], regs[:, int(imm) & 31])
            else:
                raise ValueError(f"Unknown VM opcode {op}")
        return regs[:, out_reg_start : out_reg_start + out_words]

    if prog.ndim != 3 or prog.shape[0] != b_size or prog.shape[2] != 5:
        raise ValueError(f"Expected prog shape (L, 5) or ({b_size}, L, 5), got {tuple(prog.shape)}")

    prog = prog.to(device=dev, dtype=torch.int64)
    inst_len = int(prog.shape[1])
    for t in range(inst_len):
        op = prog[:, t, 0]
        if int((op != OP_NOP).any().item()) == 0:
            continue
        dst = prog[:, t, 1].clamp(0, NUM_REGS - 1)
        a_idx = prog[:, t, 2].clamp(0, NUM_REGS - 1)
        b_raw = prog[:, t, 3]
        b_idx = b_raw.clamp(0, NUM_REGS - 1)
        imm = prog[:, t, 4]
        c_idx = imm.clamp(0, NUM_REGS - 1)

        va = regs.gather(1, a_idx.unsqueeze(1)).squeeze(1)
        vb_reg = regs.gather(1, b_idx.unsqueeze(1)).squeeze(1)
        vb = torch.where(b_raw >= 0, vb_reg, imm & MASK32)
        vc = regs.gather(1, c_idx.unsqueeze(1)).squeeze(1)

        shift = imm & 31
        rshift = (32 - shift) & 31

        res_xor = (va ^ vb) & MASK32
        res_add = (va + vb) & MASK32
        res_rotr = ((va >> shift) | ((va << rshift) & MASK32)) & MASK32
        res_shr = (va >> shift) & MASK32
        res_ch = ch32(va, vb_reg, vc)
        res_maj = maj32(va, vb_reg, vc)

        cur_dst = regs.gather(1, dst.unsqueeze(1)).squeeze(1)
        out_val = cur_dst
        out_val = torch.where(op == OP_XOR, res_xor, out_val)
        out_val = torch.where(op == OP_ADD32, res_add, out_val)
        out_val = torch.where(op == OP_ROTR, res_rotr, out_val)
        out_val = torch.where(op == OP_SHR, res_shr, out_val)
        out_val = torch.where(op == OP_CH, res_ch, out_val)
        out_val = torch.where(op == OP_MAJ, res_maj, out_val)
        regs.scatter_(1, dst.unsqueeze(1), out_val.unsqueeze(1))

    return regs[:, out_reg_start : out_reg_start + out_words]


def compile_sha256_to_vm(rounds: int, max_len: Optional[int] = None) -> torch.Tensor:
    """Compile `R`-round SHA-256 (`1 <= R <= 16`) into `(L, 5)` int64 VM bytecode.

    Registers:
      - `0..15`: free message input words `W[0..R-1]` (remaining 0)
      - `16..23`: working state `[a, b, c, d, e, f, g, h]` -> final `digest[0..7]`
      - `24..29`: temporary arithmetic registers `t0..t5`
      - `31`: reserved constant-zero register
    """
    if not (1 <= int(rounds) <= 16):
        raise ValueError(f"compile_sha256_to_vm expects 1 <= rounds <= 16, got {rounds}")

    ops = []
    # Load IV[0..7] into r16..r23 via XOR(16+j, r31=0, imm=IV[j])
    for j in range(8):
        ops.append((OP_XOR, 16 + j, 31, -1, int(SHA256_IV[j]) & MASK32))

    # Working register aliases
    ra, rb, rc, rd, re, rf, rg, rh = 16, 17, 18, 19, 20, 21, 22, 23
    t0, t1, t2, t3, t4, t5 = 24, 25, 26, 27, 28, 29
    rzero = 31

    for r in range(int(rounds)):
        # Sigma1(e) -> t0
        ops.append((OP_ROTR, t0, re, -1, 6))
        ops.append((OP_ROTR, t1, re, -1, 11))
        ops.append((OP_ROTR, t2, re, -1, 25))
        ops.append((OP_XOR, t0, t0, t1, 0))
        ops.append((OP_XOR, t0, t0, t2, 0))

        # Ch(e, f, g) -> t1
        ops.append((OP_CH, t1, re, rf, rg))

        # T1 = h + Sigma1(e) + Ch(e,f,g) + K[r] + W[r] -> t2
        ops.append((OP_ADD32, t2, rh, t0, 0))
        ops.append((OP_ADD32, t2, t2, t1, 0))
        ops.append((OP_ADD32, t2, t2, -1, int(SHA256_K[r]) & MASK32))
        ops.append((OP_ADD32, t2, t2, r, 0))

        # Sigma0(a) -> t0
        ops.append((OP_ROTR, t0, ra, -1, 2))
        ops.append((OP_ROTR, t1, ra, -1, 13))
        ops.append((OP_ROTR, t3, ra, -1, 22))
        ops.append((OP_XOR, t0, t0, t1, 0))
        ops.append((OP_XOR, t0, t0, t3, 0))

        # Maj(a, b, c) -> t1
        ops.append((OP_MAJ, t1, ra, rb, rc))

        # T2 = Sigma0(a) + Maj(a,b,c) -> t3
        ops.append((OP_ADD32, t3, t0, t1, 0))

        # new_a -> t4, new_e -> t5
        ops.append((OP_ADD32, t4, t2, t3, 0))
        ops.append((OP_ADD32, t5, rd, t2, 0))

        # Shift state: h=g, g=f, f=e, e=new_e, d=c, c=b, b=a, a=new_a
        ops.append((OP_XOR, rh, rg, rzero, 0))
        ops.append((OP_XOR, rg, rf, rzero, 0))
        ops.append((OP_XOR, rf, re, rzero, 0))
        ops.append((OP_XOR, re, t5, rzero, 0))
        ops.append((OP_XOR, rd, rc, rzero, 0))
        ops.append((OP_XOR, rc, rb, rzero, 0))
        ops.append((OP_XOR, rb, ra, rzero, 0))
        ops.append((OP_XOR, ra, t4, rzero, 0))

    # Final digest addition: digest[j] = (S_R[j] + IV[j]) & MASK32
    for j in range(8):
        ops.append((OP_ADD32, 16 + j, 16 + j, -1, int(SHA256_IV[j]) & MASK32))

    if max_len is not None:
        if len(ops) > max_len:
            raise ValueError(f"Compiled SHA-{rounds} has {len(ops)} ops > max_len={max_len}")
        while len(ops) < max_len:
            ops.append((OP_NOP, 0, 0, 0, 0))

    return torch.tensor(ops, dtype=torch.int64)


def _sample_arx_program(
    tier: int,
    n_inputs: int,
    prog_len: int,
    gen: torch.Generator,
) -> torch.Tensor:
    """Sample a single reversible-friendly or structured random ARX program of length `prog_len`.

    Curriculum tiers (`0..3`):
      - Tier 0: Pure linear GF(2) / F_2^32 circuits (`XOR`, `ROTR`, `SHR`)
      - Tier 1: Mixed ARX circuits (`XOR`, `ADD32`, `ROTR`, `SHR`)
      - Tier 2: Non-linear SHA-like ARX + local gates (`XOR`, `ADD32`, `ROTR`, `CH`, `MAJ`)
      - Tier 3: Mini-SHA Feistel-like compression blocks mixing `n_inputs` into `r16..r23`
    """
    ops = []
    # Seed state registers r16..r23 from inputs r0..(n_inputs-1) + random constants so output depends on X
    for j in range(8):
        src = j % n_inputs
        cst = int(torch.randint(0, 0x100000000, (1,), generator=gen).item())
        ops.append((OP_XOR, 16 + j, src, -1, cst))

    rot_pool = (2, 6, 7, 11, 13, 17, 18, 19, 22, 25)
    body_len = max(8, prog_len - 16)

    for step in range(body_len):
        if len(ops) >= prog_len - 8:
            break
        dst = 16 + (step % 8)
        src_in = int(torch.randint(0, n_inputs, (1,), generator=gen).item())
        b_state = 16 + int(torch.randint(0, 8, (1,), generator=gen).item())

        if tier == 0:
            choice = int(torch.randint(0, 3, (1,), generator=gen).item())
            if choice == 0:
                ops.append((OP_XOR, dst, dst, src_in, 0))
            elif choice == 1:
                sh = rot_pool[int(torch.randint(0, len(rot_pool), (1,), generator=gen).item())]
                ops.append((OP_ROTR, dst, dst, -1, sh))
            else:
                ops.append((OP_XOR, dst, dst, b_state, 0))
        elif tier == 1:
            choice = int(torch.randint(0, 4, (1,), generator=gen).item())
            if choice == 0:
                ops.append((OP_ADD32, dst, dst, src_in, 0))
            elif choice == 1:
                ops.append((OP_XOR, dst, dst, b_state, 0))
            elif choice == 2:
                sh = rot_pool[int(torch.randint(0, len(rot_pool), (1,), generator=gen).item())]
                ops.append((OP_ROTR, dst, dst, -1, sh))
            else:
                cst = int(torch.randint(0, 0x100000000, (1,), generator=gen).item())
                ops.append((OP_ADD32, dst, dst, -1, cst))
        elif tier == 2:
            choice = int(torch.randint(0, 5, (1,), generator=gen).item())
            if choice == 0:
                ops.append((OP_ADD32, dst, dst, src_in, 0))
            elif choice == 1:
                ops.append((OP_XOR, dst, dst, b_state, 0))
            elif choice == 2:
                sh = rot_pool[int(torch.randint(0, len(rot_pool), (1,), generator=gen).item())]
                ops.append((OP_ROTR, dst, dst, -1, sh))
            elif choice == 3:
                c_reg = 16 + int(torch.randint(0, 8, (1,), generator=gen).item())
                ops.append((OP_CH, dst, dst, b_state, c_reg))
            else:
                c_reg = 16 + int(torch.randint(0, 8, (1,), generator=gen).item())
                ops.append((OP_MAJ, dst, dst, b_state, c_reg))
        else:
            # Tier 3: mini SHA-like step mixing a tmp register into state
            tmp = 24 + (step % 4)
            sh = rot_pool[int(torch.randint(0, len(rot_pool), (1,), generator=gen).item())]
            c_reg = 16 + ((dst + 2) % 8)
            ops.append((OP_ROTR, tmp, b_state, -1, sh))
            if step % 2 == 0:
                ops.append((OP_CH, tmp, tmp, b_state, c_reg))
            else:
                ops.append((OP_MAJ, tmp, tmp, b_state, c_reg))
            ops.append((OP_ADD32, dst, dst, tmp, 0))
            ops.append((OP_ADD32, dst, dst, src_in, 0))

    # Mix all inputs once more near output so every x_j stays coupled in short programs
    mix_op = OP_XOR if tier == 0 else OP_ADD32
    for j in range(min(n_inputs, prog_len - len(ops))):
        ops.append((mix_op, 16 + (j % 8), 16 + (j % 8), j, 0))

    ops = ops[:prog_len]
    while len(ops) < prog_len:
        ops.append((OP_NOP, 0, 0, 0, 0))
    return torch.tensor(ops, dtype=torch.int64)


def sample_stage1_arx_curriculum_batch(
    batch_size: int,
    tier: int = 0,
    n_inputs: int = 2,
    prog_len: int = 48,
    device: torch.device | str = "cuda",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample `(prog, x_inputs, y_outputs)` for Stage 1 `f_univ` smooth ARX curriculum.

    Returns:
        prog: `(B, prog_len, 5)` int64
        x_inputs: `(B, 16)` int64 (words `n_inputs..15` are 0)
        y_outputs: `(B, 8)` int64 (`VM(prog, x_inputs)` registers `16..23`)
    """
    if generator is None:
        generator = torch.Generator(device="cpu")
    progs = [
        _sample_arx_program(tier=tier, n_inputs=n_inputs, prog_len=prog_len, gen=generator)
        for _ in range(batch_size)
    ]
    prog_t = torch.stack(progs, dim=0).to(device)
    x_free = torch.randint(
        0, 0x100000000, (batch_size, n_inputs), dtype=torch.int64, generator=generator
    )
    x_inputs = torch.zeros((batch_size, 16), dtype=torch.int64, device=device)
    x_inputs[:, :n_inputs] = x_free.to(device)
    y_outputs = execute_vm_batch(prog_t, x_inputs)
    return prog_t, x_inputs, y_outputs


__all__ = [
    "OP_NOP",
    "OP_XOR",
    "OP_ADD32",
    "OP_ROTR",
    "OP_SHR",
    "OP_CH",
    "OP_MAJ",
    "NUM_OPS",
    "NUM_REGS",
    "ARX_OP_NAMES",
    "execute_vm_batch",
    "compile_sha256_to_vm",
    "sample_stage1_arx_curriculum_batch",
]
