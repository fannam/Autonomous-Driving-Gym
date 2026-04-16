from pathlib import Path

from setuptools import find_packages, setup


PACKAGE_ROOT = Path(__file__).resolve().parent
README_PATH = PACKAGE_ROOT / "README.md"


setup(
    name="ppo-traditional-autonomous-driving",
    version="0.1.0",
    description="Traditional PPO highway driving baseline",
    long_description=README_PATH.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    package_dir={"": "."},
    packages=find_packages(where=".", include=["PPOTraditional", "PPOTraditional.*"]),
    include_package_data=True,
    python_requires=">=3.12",
    install_requires=[
        "gymnasium>=1.2.3",
        "highway-env",
        "numpy>=2.4.2",
        "PyYAML>=6.0.2",
        "torch>=2.10.0",
    ],
    entry_points={
        "console_scripts": [
            "ppo-traditional-train=PPOTraditional.scripts.train:main",
            "ppo-traditional-evaluate=PPOTraditional.scripts.evaluate:main",
        ],
    },
    zip_safe=False,
)
