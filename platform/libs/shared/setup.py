from setuptools import setup, find_packages

setup(
    name="los-ojos-shared",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.7.0",
        "aiokafka>=0.11.0",
        "redis[hiredis]>=5.0.6",
        "asyncpg>=0.29.0",
        "motor>=3.4.0",
        "structlog>=24.2.0",
    ],
)
