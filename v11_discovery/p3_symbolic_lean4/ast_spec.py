"""AST specification, dual evaluators (NumPy uint32 & Z3 BitVec32), and canonicalization."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import z3

MASK32 = 0xFFFFFFFF


def _u32(x: np.ndarray | int) -> np.ndarray | int:
    if isinstance(x, np.ndarray):
        return x.astype(np.uint32, copy=False)
    return int(x) & MASK32


def np_rotr32(x: np.ndarray, c: int) -> np.ndarray:
    c = c & 31
    if c == 0:
        return x
    xu = x.astype(np.uint32, copy=False)
    return ((xu >> np.uint32(c)) | (xu << np.uint32(32 - c))).astype(np.uint32)


def np_ch32(e: np.ndarray, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    return (e & f) ^ ((~e) & g)


def np_maj32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (a & b) ^ (a & c) ^ (b & c)


def np_Sigma0(x: np.ndarray) -> np.ndarray:
    return np_rotr32(x, 2) ^ np_rotr32(x, 13) ^ np_rotr32(x, 22)


def np_Sigma1(x: np.ndarray) -> np.ndarray:
    return np_rotr32(x, 6) ^ np_rotr32(x, 11) ^ np_rotr32(x, 25)


def np_sigma0(x: np.ndarray) -> np.ndarray:
    return np_rotr32(x, 7) ^ np_rotr32(x, 18) ^ (x >> np.uint32(3))


def np_sigma1(x: np.ndarray) -> np.ndarray:
    return np_rotr32(x, 17) ^ np_rotr32(x, 19) ^ (x >> np.uint32(10))


def z3_rotr32(x: z3.BitVecRef, c: int) -> z3.BitVecRef:
    c = c & 31
    if c == 0:
        return x
    return z3.RotateRight(x, c)


def z3_shr32(x: z3.BitVecRef, c: int) -> z3.BitVecRef:
    c = c & 31
    if c == 0:
        return x
    return z3.LShR(x, c)


def z3_ch32(e: z3.BitVecRef, f: z3.BitVecRef, g: z3.BitVecRef) -> z3.BitVecRef:
    return (e & f) ^ ((~e) & g)


def z3_maj32(a: z3.BitVecRef, b: z3.BitVecRef, c: z3.BitVecRef) -> z3.BitVecRef:
    return (a & b) ^ (a & c) ^ (b & c)


def z3_Sigma0(x: z3.BitVecRef) -> z3.BitVecRef:
    return z3_rotr32(x, 2) ^ z3_rotr32(x, 13) ^ z3_rotr32(x, 22)


def z3_Sigma1(x: z3.BitVecRef) -> z3.BitVecRef:
    return z3_rotr32(x, 6) ^ z3_rotr32(x, 11) ^ z3_rotr32(x, 25)


def z3_sigma0(x: z3.BitVecRef) -> z3.BitVecRef:
    return z3_rotr32(x, 7) ^ z3_rotr32(x, 18) ^ z3_shr32(x, 3)


def z3_sigma1(x: z3.BitVecRef) -> z3.BitVecRef:
    return z3_rotr32(x, 17) ^ z3_rotr32(x, 19) ^ z3_shr32(x, 10)


# --- AST Constructors (strictly matching SPEC.md Section 3) ---

def Var(name: str) -> Dict[str, Any]:
    return {"op": "Var", "name": name}


def Const(val: int) -> Dict[str, Any]:
    return {"op": "Const", "val": int(val) & MASK32}


def Add(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Add", "args": [l, r]}


def Sub(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Sub", "args": [l, r]}


def Neg(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Neg", "args": [x]}


def And(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "And", "args": [l, r]}


def Or(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Or", "args": [l, r]}


def Xor(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Xor", "args": [l, r]}


def Not(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Not", "args": [x]}


def Rotr(x: Dict[str, Any], c: int) -> Dict[str, Any]:
    return {"op": "Rotr", "c": int(c) & 31, "args": [x]}


def Shr(x: Dict[str, Any], c: int) -> Dict[str, Any]:
    return {"op": "Shr", "c": int(c) & 31, "args": [x]}


def Shl(x: Dict[str, Any], c: int) -> Dict[str, Any]:
    return {"op": "Shl", "c": int(c) & 31, "args": [x]}


def Sigma0(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Sigma0", "args": [x]}


def Sigma1(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Sigma1", "args": [x]}


def sigma0(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "sigma0", "args": [x]}


def sigma1(x: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "sigma1", "args": [x]}


def Ch(e: Dict[str, Any], f: Dict[str, Any], g: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Ch", "args": [e, f, g]}


def Maj(a: Dict[str, Any], b: Dict[str, Any], c: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Maj", "args": [a, b, c]}


# Cond constructors
def CondTrue() -> Dict[str, Any]:
    return {"op": "True"}


def CondEq(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "Eq", "args": [l, r]}


def CondMaskZero(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "MaskZero", "args": [a, b]}


def CondMaskEq(a: Dict[str, Any], b: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "MaskEq", "args": [a, b, m]}


def CondBitEq(x: Dict[str, Any], k: int, bit01: int) -> Dict[str, Any]:
    return {"op": "BitEq", "k": int(k) & 31, "bit": int(bit01) & 1, "args": [x]}


def CondAnd(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "AndCond", "args": [l, r]}


# --- Metrics & Utilities ---

def ast_stats(node: Dict[str, Any]) -> Tuple[int, int]:
    """Return (node_count, depth)."""
    args = node.get("args", [])
    if not args:
        return 1, 1
    counts, depths = zip(*(ast_stats(a) for a in args))
    return 1 + sum(counts), 1 + max(depths)


def collect_vars(node: Dict[str, Any], out: Optional[Set[str]] = None) -> Set[str]:
    if out is None:
        out = set()
    if node["op"] == "Var":
        out.add(node["name"])
    for a in node.get("args", []):
        collect_vars(a, out)
    return out


def canonical_fingerprint(
    track: str,
    form: str,
    cond_ast: Dict[str, Any],
    mask_ast: Dict[str, Any],
    lhs_ast: Dict[str, Any],
    rhs_ast: Dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "track": track,
            "form": form,
            "cond": cond_ast,
            "mask": mask_ast,
            "lhs": lhs_ast,
            "rhs": rhs_ast,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def ast_tokens(node: Dict[str, Any], max_len: int = 64) -> List[str]:
    tokens: List[str] = []

    def _walk(n: Dict[str, Any]) -> None:
        if len(tokens) >= max_len:
            return
        op = n["op"]
        if op == "Var":
            tokens.append(f"Var:{n['name'].split('.')[0]}")
        elif op == "Const":
            tokens.append("Const")
        else:
            tokens.append(op)
        for a in n.get("args", []):
            _walk(a)

    _walk(node)
    return tokens


def pretty_expr(node: Dict[str, Any]) -> str:
    op = node["op"]
    if op == "Var":
        return node["name"]
    if op == "Const":
        v = node["val"]
        return f"0x{v:08x}" if v > 9 else str(v)
    args = node.get("args", [])
    if op == "Add":
        return f"({pretty_expr(args[0])} + {pretty_expr(args[1])})"
    if op == "Sub":
        return f"({pretty_expr(args[0])} - {pretty_expr(args[1])})"
    if op == "Neg":
        return f"(-{pretty_expr(args[0])})"
    if op == "And":
        return f"({pretty_expr(args[0])} & {pretty_expr(args[1])})"
    if op == "Or":
        return f"({pretty_expr(args[0])} | {pretty_expr(args[1])})"
    if op == "Xor":
        return f"({pretty_expr(args[0])} ^ {pretty_expr(args[1])})"
    if op == "Not":
        return f"(~{pretty_expr(args[0])})"
    if op in ("Rotr", "Shr", "Shl"):
        return f"{op.lower()}32({pretty_expr(args[0])}, {node['c']})"
    if op in ("Sigma0", "Sigma1", "sigma0", "sigma1"):
        return f"{op}({pretty_expr(args[0])})"
    if op in ("Ch", "Maj"):
        return f"{op}({pretty_expr(args[0])}, {pretty_expr(args[1])}, {pretty_expr(args[2])})"
    raise ValueError(f"Unknown Expr32 op: {op}")


def pretty_cond(node: Dict[str, Any]) -> str:
    op = node["op"]
    if op == "True":
        return "True"
    args = node.get("args", [])
    if op == "Eq":
        return f"({pretty_expr(args[0])} == {pretty_expr(args[1])})"
    if op == "MaskZero":
        return f"(({pretty_expr(args[0])} & {pretty_expr(args[1])}) == 0)"
    if op == "MaskEq":
        return f"(({pretty_expr(args[0])} & {pretty_expr(args[2])}) == ({pretty_expr(args[1])} & {pretty_expr(args[2])}))"
    if op == "BitEq":
        return f"((({pretty_expr(args[0])} >> {node['k']}) & 1) == {node['bit']})"
    if op == "AndCond":
        return f"({pretty_cond(args[0])} && {pretty_cond(args[1])})"
    raise ValueError(f"Unknown Cond op: {op}")


# --- Vectorized NumPy Evaluation ---

def eval_expr_np(
    node: Dict[str, Any],
    env: Dict[str, np.ndarray],
    batch_size: int,
    cache: Optional[Dict[int, np.ndarray]] = None,
) -> np.ndarray:
    if cache is None:
        cache = {}
    nid = id(node)
    if nid in cache:
        return cache[nid]
    op = node["op"]
    if op == "Var":
        res = env[node["name"]]
    elif op == "Const":
        res = np.full((batch_size,), np.uint32(node["val"]), dtype=np.uint32)
    else:
        args = node.get("args", [])
        if op == "Add":
            res = (eval_expr_np(args[0], env, batch_size, cache) + eval_expr_np(args[1], env, batch_size, cache)).astype(np.uint32)
        elif op == "Sub":
            res = (eval_expr_np(args[0], env, batch_size, cache) - eval_expr_np(args[1], env, batch_size, cache)).astype(np.uint32)
        elif op == "Neg":
            res = (-eval_expr_np(args[0], env, batch_size, cache)).astype(np.uint32)
        elif op == "And":
            res = eval_expr_np(args[0], env, batch_size, cache) & eval_expr_np(args[1], env, batch_size, cache)
        elif op == "Or":
            res = eval_expr_np(args[0], env, batch_size, cache) | eval_expr_np(args[1], env, batch_size, cache)
        elif op == "Xor":
            res = eval_expr_np(args[0], env, batch_size, cache) ^ eval_expr_np(args[1], env, batch_size, cache)
        elif op == "Not":
            res = (~eval_expr_np(args[0], env, batch_size, cache)).astype(np.uint32)
        elif op == "Rotr":
            res = np_rotr32(eval_expr_np(args[0], env, batch_size, cache), node["c"])
        elif op == "Shr":
            c = node["c"]
            x = eval_expr_np(args[0], env, batch_size, cache)
            res = x if c == 0 else (x >> np.uint32(c)).astype(np.uint32)
        elif op == "Shl":
            c = node["c"]
            x = eval_expr_np(args[0], env, batch_size, cache)
            res = x if c == 0 else (x << np.uint32(c)).astype(np.uint32)
        elif op == "Sigma0":
            res = np_Sigma0(eval_expr_np(args[0], env, batch_size, cache))
        elif op == "Sigma1":
            res = np_Sigma1(eval_expr_np(args[0], env, batch_size, cache))
        elif op == "sigma0":
            res = np_sigma0(eval_expr_np(args[0], env, batch_size, cache))
        elif op == "sigma1":
            res = np_sigma1(eval_expr_np(args[0], env, batch_size, cache))
        elif op == "Ch":
            res = np_ch32(
                eval_expr_np(args[0], env, batch_size, cache),
                eval_expr_np(args[1], env, batch_size, cache),
                eval_expr_np(args[2], env, batch_size, cache),
            )
        elif op == "Maj":
            res = np_maj32(
                eval_expr_np(args[0], env, batch_size, cache),
                eval_expr_np(args[1], env, batch_size, cache),
                eval_expr_np(args[2], env, batch_size, cache),
            )
        else:
            raise ValueError(f"Unknown Expr32 op: {op}")
    cache[nid] = res
    return res


def eval_cond_np(node: Dict[str, Any], env: Dict[str, np.ndarray], batch_size: int) -> np.ndarray:
    op = node["op"]
    if op == "True":
        return np.ones((batch_size,), dtype=bool)
    args = node.get("args", [])
    if op == "Eq":
        return eval_expr_np(args[0], env, batch_size) == eval_expr_np(args[1], env, batch_size)
    if op == "MaskZero":
        return (eval_expr_np(args[0], env, batch_size) & eval_expr_np(args[1], env, batch_size)) == np.uint32(0)
    if op == "MaskEq":
        m = eval_expr_np(args[2], env, batch_size)
        return (eval_expr_np(args[0], env, batch_size) & m) == (eval_expr_np(args[1], env, batch_size) & m)
    if op == "BitEq":
        x = eval_expr_np(args[0], env, batch_size)
        return ((x >> np.uint32(node["k"])) & np.uint32(1)) == np.uint32(node["bit"])
    if op == "AndCond":
        return eval_cond_np(args[0], env, batch_size) & eval_cond_np(args[1], env, batch_size)
    raise ValueError(f"Unknown Cond op: {op}")


# --- Z3 BitVec32 Compilation ---

def compile_expr_z3(
    node: Dict[str, Any],
    env: Dict[str, z3.BitVecRef],
    cache: Optional[Dict[int, z3.BitVecRef]] = None,
) -> z3.BitVecRef:
    if cache is None:
        cache = {}
    if id(node) in cache:
        return cache[id(node)]
    op = node["op"]
    if op == "Var":
        res = env[node["name"]]
    elif op == "Const":
        res = z3.BitVecVal(int(node["val"]) & MASK32, 32)
    else:
        args = [compile_expr_z3(a, env, cache) for a in node.get("args", [])]
        if op == "Add":
            res = args[0] + args[1]
        elif op == "Sub":
            res = args[0] - args[1]
        elif op == "Neg":
            res = -args[0]
        elif op == "And":
            res = args[0] & args[1]
        elif op == "Or":
            res = args[0] | args[1]
        elif op == "Xor":
            res = args[0] ^ args[1]
        elif op == "Not":
            res = ~args[0]
        elif op == "Rotr":
            res = z3_rotr32(args[0], node["c"])
        elif op == "Shr":
            res = z3_shr32(args[0], node["c"])
        elif op == "Shl":
            c = node["c"]
            res = args[0] if c == 0 else (args[0] << c)
        elif op == "Sigma0":
            res = z3_Sigma0(args[0])
        elif op == "Sigma1":
            res = z3_Sigma1(args[0])
        elif op == "sigma0":
            res = z3_sigma0(args[0])
        elif op == "sigma1":
            res = z3_sigma1(args[0])
        elif op == "Ch":
            res = z3_ch32(args[0], args[1], args[2])
        elif op == "Maj":
            res = z3_maj32(args[0], args[1], args[2])
        else:
            raise ValueError(f"Unknown Expr32 op: {op}")
    cache[id(node)] = res
    return res


def compile_cond_z3(node: Dict[str, Any], env: Dict[str, z3.BitVecRef]) -> z3.BoolRef:
    op = node["op"]
    if op == "True":
        return z3.BoolVal(True)
    args = node.get("args", [])
    if op == "Eq":
        return compile_expr_z3(args[0], env) == compile_expr_z3(args[1], env)
    if op == "MaskZero":
        return (compile_expr_z3(args[0], env) & compile_expr_z3(args[1], env)) == z3.BitVecVal(0, 32)
    if op == "MaskEq":
        m = compile_expr_z3(args[2], env)
        return (compile_expr_z3(args[0], env) & m) == (compile_expr_z3(args[1], env) & m)
    if op == "BitEq":
        x = compile_expr_z3(args[0], env)
        return ((z3.LShR(x, node["k"])) & z3.BitVecVal(1, 32)) == z3.BitVecVal(node["bit"], 32)
    if op == "AndCond":
        return z3.And(compile_cond_z3(args[0], env), compile_cond_z3(args[1], env))
    raise ValueError(f"Unknown Cond op: {op}")
