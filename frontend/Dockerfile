FROM node:alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./

RUN npm ci

COPY frontend ./

ARG VITE_API_ENDPOINT=${VITE_API_ENDPOINT}

RUN npm run build

FROM node:alpine

COPY --from=build /app/build ./

ENV NODE_ENV=production PORT=3000

EXPOSE 3000

CMD ["node", "index.js"]