
# Stage 1: Build
FROM node:20-alpine AS builder

WORKDIR /app

# Install deps
COPY package.json ./
RUN npm install --legacy-peer-deps

# Build
COPY . .
ARG VITE_WS_URL=ws://localhost:8005/ws
ARG VITE_API_URL=http://localhost:8005
ENV VITE_WS_URL=$VITE_WS_URL
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Stage 2: Serve with nginx
FROM nginx:1.25-alpine

COPY --from=builder /app/dist /usr/share/nginx/html

# Nginx config for SPA routing
RUN printf 'server {\n\
  listen 3000;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  gzip on;\n\
  gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
  location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {\n\
    expires 1y;\n\
    add_header Cache-Control "public, immutable";\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
