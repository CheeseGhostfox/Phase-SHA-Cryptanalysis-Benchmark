"""Exact SHA-256 & 128-round BTC Double-SHA256 reference engine and dependency graph.

All arithmetic is exact 32-bit unsigned (mod 2^32). Cross-checked against
``g:/SHA-LM/src/sha256_ref.py`` and Python's ``hashlib.sha256`` (xor_diff == 0).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

MASK32: int = 0xFFFFFFFF


def _first_n_primes(n: int) -> Tuple[int, ...]:
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        limit = math.isqrt(candidate)
        if all(candidate % p != 0 for p in primes if p <= limit):
            primes.append(candidate)
        candidate += 1
    return tuple(primes)


def _icbrt(n: int) -> int:
    if n <= 0:
        return 0
    x = 1 << ((n.bit_length() + 2) // 3)
    while True:
        y = (2 * x + n // (x * x)) // 3
        if y >= x:
            return x
        x = y


def _derive_sha256_constants() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    primes = _first_n_primes(64)
    iv = tuple((math.isqrt(p << 64)) & MASK32 for p in primes[:8])
    k = tuple((_icbrt(p << 96)) & MASK32 for p in primes)
    return iv, k


SHA256_IV, SHA256_K = _derive_sha256_constants()


def rotr32(x: int, n: int) -> int:
    x &= MASK32
    if n == 0:
        return x
    return ((x >> n) | ((x & ((1 << n) - 1)) << (32 - n))) & MASK32


def shr32(x: int, n: int) -> int:
    return (x & MASK32) >> n


def ch32(e: int, f: int, g: int) -> int:
    e &= MASK32
    f &= MASK32
    g &= MASK32
    return ((e & f) ^ ((~e & MASK32) & g)) & MASK32


def maj32(a: int, b: int, c: int) -> int:
    a &= MASK32
    b &= MASK32
    c &= MASK32
    return ((a & b) ^ (a & c) ^ (b & c)) & MASK32


def sigma0_upper(x: int) -> int:
    return (rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22)) & MASK32


def sigma1_upper(x: int) -> int:
    return (rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25)) & MASK32


def sigma0_lower(x: int) -> int:
    return (rotr32(x, 7) ^ rotr32(x, 18) ^ shr32(x, 3)) & MASK32


def sigma1_lower(x: int) -> int:
    return (rotr32(x, 17) ^ rotr32(x, 19) ^ shr32(x, 10)) & MASK32


def expand_message_schedule(w16: Sequence[int], rounds: int = 64) -> List[int]:
    if len(w16) != 16:
        raise ValueError(f"Expected 16 words, got {len(w16)}")
    if not (1 <= rounds <= 64):
        raise ValueError(f"rounds must be in [1, 64], got {rounds}")
    if rounds <= 16:
        return [int(w) & MASK32 for w in w16[:rounds]]
    w = [int(x) & MASK32 for x in w16]
    for t in range(16, rounds):
        wt = (
            sigma1_lower(w[t - 2])
            + w[t - 7]
            + sigma0_lower(w[t - 15])
            + w[t - 16]
        ) & MASK32
        w.append(wt)
    return w


def step_forward(
    state: Tuple[int, ...], w_t: int, t: int
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Advance 8-word state (a,b,c,d,e,f,g,h) by one SHA-256 round t."""
    a, b, c, d, e, f, g, h = (x & MASK32 for x in state)
    t1 = (
        h + sigma1_upper(e) + ch32(e, f, g) + SHA256_K[t] + (w_t & MASK32)
    ) & MASK32
    t2 = (sigma0_upper(a) + maj32(a, b, c)) & MASK32
    new_a = (t1 + t2) & MASK32
    new_e = (d + t1) & MASK32
    return (new_a, a, b, c, new_e, e, f, g)


def step_backward(
    next_state: Tuple[int, ...], w_t: int, t: int
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Exact algebraic inverse of step_forward given w_t and round index t."""
    na, nb, nc, nd, ne, nf, ng, nh = (x & MASK32 for x in next_state)
    a, b, c = nb, nc, nd
    e, f, g = nf, ng, nh
    t2 = (sigma0_upper(a) + maj32(a, b, c)) & MASK32
    t1 = (na - t2) & MASK32
    d = (ne - t1) & MASK32
    h = (
        t1 - sigma1_upper(e) - ch32(e, f, g) - SHA256_K[t] - (w_t & MASK32)
    ) & MASK32
    return (a, b, c, d, e, f, g, h)


@dataclass
class CompressionTrace:
    """Complete trace using SHA-LM two-track indexing: A_full[k+3] = A_k."""

    iv: Tuple[int, ...]
    A_full: List[int]
    E_full: List[int]
    W_full: List[int]
    S_R: Tuple[int, ...]
    digest: Tuple[int, ...]


def compress_trace(
    w16: Sequence[int],
    iv: Sequence[int] = SHA256_IV,
    rounds: int = 64,
) -> CompressionTrace:
    """Run exact SHA-256 compression on 16-word block from arbitrary IV."""
    if len(iv) != 8:
        raise ValueError(f"Expected 8 IV words, got {len(iv)}")
    iv_t = tuple(int(x) & MASK32 for x in iv)
    w_full = expand_message_schedule(w16, rounds=rounds)

    # A_full[k + 3] = A_k, E_full[k + 3] = E_k for k in -3..rounds
    a_cols = [iv_t[3], iv_t[2], iv_t[1], iv_t[0]]
    e_cols = [iv_t[7], iv_t[6], iv_t[5], iv_t[4]]

    state = iv_t
    for t in range(rounds):
        state = step_forward(state, w_full[t], t)
        a_cols.append(state[0])
        e_cols.append(state[4])

    s_r = state
    digest = tuple((s_r[i] + iv_t[i]) & MASK32 for i in range(8))
    return CompressionTrace(
        iv=iv_t,
        A_full=a_cols,
        E_full=e_cols,
        W_full=w_full,
        S_R=s_r,
        digest=digest,
    )


def pad_bytes_to_blocks(msg: bytes) -> List[List[int]]:
    """Standard FIPS 180-4 SHA-256 padding into 512-bit (16-word) blocks."""
    bit_len = len(msg) * 8
    buf = bytearray(msg)
    buf.append(0x80)
    while (len(buf) % 64) != 56:
        buf.append(0)
    buf.extend(bit_len.to_bytes(8, byteorder="big"))
    blocks: List[List[int]] = []
    for off in range(0, len(buf), 64):
        blk = [
            int.from_bytes(buf[off + 4 * i : off + 4 * (i + 1)], "big")
            for i in range(16)
        ]
        blocks.append(blk)
    return blocks


def sha256_bytes(msg: bytes) -> Tuple[int, ...]:
    iv = SHA256_IV
    for blk in pad_bytes_to_blocks(msg):
        iv = compress_trace(blk, iv=iv, rounds=64).digest
    return iv


def words_to_bytes(words: Sequence[int]) -> bytes:
    return b"".join((int(w) & MASK32).to_bytes(4, "big") for w in words)


def clz_word(w: int) -> int:
    w &= MASK32
    return 32 if w == 0 else 32 - w.bit_length()


def clz_digest_be(digest: Sequence[int]) -> int:
    """Count leading zero bits across big-endian 256-bit digest."""
    total = 0
    for w in digest:
        c = clz_word(w)
        total += c
        if c < 32:
            break
    return total


def clz_digest_btc_le(digest: Sequence[int]) -> int:
    """Count leading zero bits in Bitcoin little-endian block-hash order."""
    raw = words_to_bytes(digest)[::-1]
    val = int.from_bytes(raw, "big")
    return 256 if val == 0 else 256 - val.bit_length()


@dataclass
class BTCDoubleSHA256Trace:
    """Full trace for 80-byte Bitcoin header Double-SHA256."""

    hdr_block1: CompressionTrace
    hdr_block2: CompressionTrace  # 64 rounds: W[4]=0x80000000, W[5..14]=0, W[15]=640
    outer_block: CompressionTrace  # 64 rounds: W'[0..7]=H_inner, W'[8]=0x80000000, W'[15]=256
    final_digest: Tuple[int, ...]


def btc_double_sha256_trace(header_80b: bytes) -> BTCDoubleSHA256Trace:
    """Exact 80-byte Bitcoin block header Double-SHA256 with all block traces."""
    if len(header_80b) != 80:
        raise ValueError(f"Expected 80-byte BTC header, got {len(header_80b)} bytes")
    blocks = pad_bytes_to_blocks(header_80b)
    if len(blocks) != 2:
        raise RuntimeError(f"Expected 2 padded blocks for 80B header, got {len(blocks)}")
    assert blocks[1][4] == 0x80000000
    assert all(blocks[1][i] == 0 for i in range(5, 15))
    assert blocks[1][15] == 640

    tr1 = compress_trace(blocks[0], iv=SHA256_IV, rounds=64)
    tr2 = compress_trace(blocks[1], iv=tr1.digest, rounds=64)

    outer_w16 = list(tr2.digest) + [0x80000000, 0, 0, 0, 0, 0, 0, 256]
    tr_outer = compress_trace(outer_w16, iv=SHA256_IV, rounds=64)
    return BTCDoubleSHA256Trace(
        hdr_block1=tr1,
        hdr_block2=tr2,
        outer_block=tr_outer,
        final_digest=tr_outer.digest,
    )


def double_sha256_single_block_trace(
    w16: Sequence[int], iv1: Sequence[int] = SHA256_IV, rounds1: int = 64, rounds2: int = 64
) -> Tuple[CompressionTrace, CompressionTrace]:
    """128-round (or r1+r2) Double-SHA256 where Block 1 is 16 words -> H_inner -> Block 2."""
    tr1 = compress_trace(w16, iv=iv1, rounds=rounds1)
    outer_w16 = list(tr1.digest) + [0x80000000, 0, 0, 0, 0, 0, 0, 256]
    tr2 = compress_trace(outer_w16, iv=SHA256_IV, rounds=rounds2)
    return tr1, tr2


# ---------------------------------------------------------------------------
# Message Expansion & State Dependency Graph (64-round & 128-round BTC)
# ---------------------------------------------------------------------------

GraphMode = Literal["sha256_64", "btc_hdr2_64", "btc_double_128", "double_512_128"]


@dataclass
class MsgWordNode:
    """Node in the message-schedule dependency graph."""

    block: int  # 0 for first active block, 1 for outer block
    round_idx: int  # 0..63
    global_idx: int  # round_idx (block 0) or 64 + round_idx (block 1)
    name: str
    is_const: bool = False
    const_val: Optional[int] = None
    # Dependencies: W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]
    # Tuple of (source_name, edge_type) where edge_type in {"lin", "sigma0", "sigma1", "splice"}
    deps: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class SHA256DependencyGraph:
    """Exact 64-round or 128-round message-schedule & compression dependency DAG."""

    mode: GraphMode
    num_rounds: int
    msg_nodes: Dict[str, MsgWordNode]
    msg_order: List[str]
    constant_words: Dict[str, int]
    splice_links: Dict[str, List[str]]  # e.g. W1[0..7] <- Block 0 end state + midstate


def _propagate_msg_block(
    block_idx: int,
    prefix: str,
    rounds: int,
    init_consts: Dict[int, int],
) -> Tuple[Dict[str, MsgWordNode], List[str]]:
    nodes: Dict[str, MsgWordNode] = {}
    order: List[str] = []
    for t in range(rounds):
        name = f"{prefix}[{t}]"
        gidx = block_idx * 64 + t
        if t < 16:
            is_c = t in init_consts
            c_val = init_consts.get(t)
            node = MsgWordNode(
                block=block_idx,
                round_idx=t,
                global_idx=gidx,
                name=name,
                is_const=is_c,
                const_val=c_val,
                deps=[],
            )
        else:
            d_m2 = f"{prefix}[{t-2}]"
            d_m7 = f"{prefix}[{t-7}]"
            d_m15 = f"{prefix}[{t-15}]"
            d_m16 = f"{prefix}[{t-16}]"
            deps = [
                (d_m2, "sigma1"),
                (d_m7, "lin"),
                (d_m15, "sigma0"),
                (d_m16, "lin"),
            ]
            if all(nodes[d].is_const for d, _ in deps):
                v2 = nodes[d_m2].const_val
                v7 = nodes[d_m7].const_val
                v15 = nodes[d_m15].const_val
                v16 = nodes[d_m16].const_val
                assert v2 is not None and v7 is not None and v15 is not None and v16 is not None
                c_val = (sigma1_lower(v2) + v7 + sigma0_lower(v15) + v16) & MASK32
                is_c = True
            else:
                c_val = None
                is_c = False
            node = MsgWordNode(
                block=block_idx,
                round_idx=t,
                global_idx=gidx,
                name=name,
                is_const=is_c,
                const_val=c_val,
                deps=deps,
            )
        nodes[name] = node
        order.append(name)
    return nodes, order


def build_dependency_graph(
    mode: GraphMode = "btc_double_128",
    rounds_block0: int = 64,
    rounds_block1: int = 64,
) -> SHA256DependencyGraph:
    """Build exact message schedule & inter-block splice dependency graph."""
    if mode == "sha256_64":
        nodes, order = _propagate_msg_block(0, "W", rounds_block0, {})
        consts = {n: nd.const_val for n, nd in nodes.items() if nd.is_const and nd.const_val is not None}
        return SHA256DependencyGraph(
            mode=mode,
            num_rounds=rounds_block0,
            msg_nodes=nodes,
            msg_order=order,
            constant_words=consts,
            splice_links={},
        )

    if mode == "btc_hdr2_64":
        init_c = {4: 0x80000000, **{i: 0 for i in range(5, 15)}, 15: 640}
        nodes, order = _propagate_msg_block(0, "W", rounds_block0, init_c)
        consts = {n: nd.const_val for n, nd in nodes.items() if nd.is_const and nd.const_val is not None}
        return SHA256DependencyGraph(
            mode=mode,
            num_rounds=rounds_block0,
            msg_nodes=nodes,
            msg_order=order,
            constant_words=consts,
            splice_links={},
        )

    if mode in ("btc_double_128", "double_512_128"):
        init_c0 = (
            {4: 0x80000000, **{i: 0 for i in range(5, 15)}, 15: 640}
            if mode == "btc_double_128"
            else {}
        )
        n0, o0 = _propagate_msg_block(0, "W0", rounds_block0, init_c0)
        init_c1 = {8: 0x80000000, **{i: 0 for i in range(9, 15)}, 15: 256}
        n1, o1 = _propagate_msg_block(1, "W1", rounds_block1, init_c1)

        # Link W1[0..7] to Block 0 output registers A_{R0},..,A_{R0-3}, E_{R0},..,E_{R0-3}
        splice_links: Dict[str, List[str]] = {
            f"W1[{i}]": [f"B0_H[{i}]"] for i in range(8)
        }
        for i in range(8):
            n1[f"W1[{i}]"].deps.append((f"B0_H[{i}]", "splice"))

        all_nodes = {**n0, **n1}
        all_order = o0 + o1
        consts = {
            n: nd.const_val
            for n, nd in all_nodes.items()
            if nd.is_const and nd.const_val is not None
        }
        return SHA256DependencyGraph(
            mode=mode,
            num_rounds=rounds_block0 + rounds_block1,
            msg_nodes=all_nodes,
            msg_order=all_order,
            constant_words=consts,
            splice_links=splice_links,
        )

    raise ValueError(f"Unsupported graph mode: {mode}")
