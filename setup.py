"""
Setup configuration for VoiceOps AI package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="voiceops-ai",
    version="1.0.0",
    author="VoiceOps AI Team",
    author_email="team@voiceops-ai.com",
    description="Enterprise Voice-Driven Operational Automation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/voiceops-ai",
    packages=find_packages(exclude=["tests", "notebooks", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "voiceops-ai=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
