# ARD Skill Hub

Skill management system for ARD Character Classification System.

## ✨ Features

- **Skill Registration**: Register and manage custom skills
- **Version Management**: Multi-version skill support
- **Category Browsing**: Browse skills by category
- **One-click Installation**: Simple skill installation
- **CLI Tool**: Command-line management
- **RESTful API**: Complete API interface

## 🚀 Quick Start

### Install CLI

```bash
# Install ARD CLI
bash sh/install.sh

# Verify installation
ardc --help
```

### Start Service

```bash
# Using systemd
sudo systemctl start ardc-api

# Or manual start
uvicorn ardc.api.main:app --host 0.0.0.0 --port 8000
```

### API Access

- **API**: `http://localhost:8000/api/`
- **Docs**: `http://localhost:8000/docs`

## 🌐 CLI Commands

```bash
# Skill Management
ardc skill list              # List all skills
ardc skill install <id>      # Install skill
ardc skill uninstall <id>    # Uninstall skill
ardc skill register          # Register new skill
ardc skill update <id>       # Update skill

# System Management
ardc system status           # Check system status
ardc system upgrade          # Upgrade system
ardc system clean            # Clean cache

# Search
ardc search <keyword>        # Search skills
```

## 📁 Project Structure

```
skillhub/
├── ardc/                    # ARD CLI Core
│   ├── api/                 # RESTful API
│   ├── cli/                 # Command Line Interface
│   └── core/                # Core Modules
├── web/                     # Frontend
├── conf/                    # Configuration
├── sh/                      # Shell Scripts
└── docs/                    # Documentation
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/skills` | GET | Get skill list |
| `/api/skills/{id}` | GET | Get skill detail |
| `/api/skills` | POST | Register skill |
| `/api/skills/{id}/install` | POST | Install skill |
| `/api/skills/{id}/uninstall` | POST | Uninstall skill |
| `/api/health` | GET | Health check |

## 🔧 Configuration

Environment Variables:

```bash
export JWT_SECRET_KEY="your-secret-key"
export ALLOWED_ORIGINS="http://localhost:3000"
export ARD_C_DATA_DIR="~/.ardc"
```

## 📚 Documentation

Documentation:
- `docs/DEPLOYMENT.md` - Deployment guide
- `docs/API.md` - API documentation
- `docs/CLI.md` - CLI manual

## 📄 License

MIT License

---

**Version**: v1.0 | **Last Updated**: May 2026