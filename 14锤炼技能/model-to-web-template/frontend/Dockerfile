# Frontend Dockerfile
FROM node:20-slim

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

COPY frontend .

ENV NEXT_PUBLIC_API_BASE=http://backend:8000
EXPOSE 3000
CMD ["npm", "run", "dev"]
