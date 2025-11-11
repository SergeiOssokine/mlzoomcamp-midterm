FROM python:3.13-slim
RUN pip install uv
COPY ["pyproject.toml","uv.lock","./"]
RUN uv sync --frozen --no-dev --no-install-project
COPY ["predict.py","seeds_classifier.pkl","./"]
CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
EXPOSE 9696