"""Encrypt JSONL at rest."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


def encrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Encrypt a JSONL file in place line by line.
    
    Each line is encrypted separately so the file can still be processed
    line-by-line without loading the entire file into memory.
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library is required for encryption")
        
    if not key:
        raise ValueError("Encryption key is required")
        
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
        
    fernet = Fernet(key)
    temp_path = p.with_suffix(p.suffix + ".enc.tmp")
    
    try:
        with open(p, "rb") as f_in, open(temp_path, "wb") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                encrypted_line = fernet.encrypt(line.strip())
                f_out.write(encrypted_line + b"\n")
                
        # Replace original file with encrypted version
        os.replace(temp_path, p)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e


def decrypt_jsonl(path: Path | str, *, key: bytes | None = None) -> None:
    """Decrypt a JSONL file in place line by line."""
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library is required for decryption")
        
    if not key:
        raise ValueError("Decryption key is required")
        
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
        
    fernet = Fernet(key)
    temp_path = p.with_suffix(p.suffix + ".dec.tmp")
    
    try:
        with open(p, "rb") as f_in, open(temp_path, "wb") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                decrypted_line = fernet.decrypt(line.strip())
                f_out.write(decrypted_line + b"\n")
                
        # Replace original file with decrypted version
        os.replace(temp_path, p)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise e
