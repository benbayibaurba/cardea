FROM python:3.11-slim
WORKDIR /cardea
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY app ./app
RUN pip install --no-cache-dir -e ".[app]"
EXPOSE 7860
CMD ["python", "app/demo.py"]
