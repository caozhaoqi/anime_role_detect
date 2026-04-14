#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from src.backend.api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.backend.api.app:app", host="127.0.0.1", port=8001, reload=True)
