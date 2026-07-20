# Stage 1: Build React application
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files first for cached dependency layering
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Serve compiled assets with Nginx proxy server
FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
