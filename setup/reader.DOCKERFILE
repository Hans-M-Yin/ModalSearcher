ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

ARG REPO_URL=https://github.com/jina-ai/reader.git
ARG REPO_REF=main
ARG NODE_VERSION=24.14.1

ENV DEBIAN_FRONTEND=noninteractive
ENV PORT=8080
ENV NODE_COMPILE_CACHE=node_modules
ENV OVERRIDE_CHROME_EXECUTABLE_PATH=/usr/bin/google-chrome-stable

USER root
WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      xz-utils \
      gnupg \
      sudo \
      bash \
      coreutils \
      procps \
      build-essential \
      python3 \
      pkg-config && \
    install -d -m 0755 /etc/apt/keyrings && \
    curl -fsSL https://dl.google.com/linux/linux_signing_key.pub \
      | gpg --dearmor -o /etc/apt/keyrings/google-linux.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/google-linux.gpg] http://dl.google.com/linux/chrome/deb/ stable main" \
      > /etc/apt/sources.list.d/google-chrome.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libreoffice \
      google-chrome-stable \
      fonts-noto-cjk-extra \
      fonts-noto-color-emoji \
      fonts-liberation \
      fonts-ipafont-gothic \
      fonts-wqy-zenhei \
      fonts-thai-tlwg \
      fonts-kacst \
      fonts-freefont-ttf \
      libxss1 \
      zstd \
      libc++-dev && \
    rm -rf /var/lib/apt/lists/*

RUN arch="$(dpkg --print-architecture)" && \
    case "$arch" in \
      amd64) node_arch="x64" ;; \
      arm64) node_arch="arm64" ;; \
      *) echo "Unsupported architecture: $arch" && exit 1 ;; \
    esac && \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" && \
    tar -xJf "node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" -C /usr/local --strip-components=1 && \
    rm "node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" && \
    node --version && \
    npm --version

RUN git clone --depth 1 --branch "${REPO_REF}" "${REPO_URL}" /workspace/reader

WORKDIR /workspace/reader

RUN npm ci

RUN mkdir -p licensed && \
    curl -fsSL -o licensed/GeoLite2-City.mmdb \
      https://raw.githubusercontent.com/P3TERX/GeoLite.mmdb/download/GeoLite2-City.mmdb && \
    curl -fsSL -o licensed/geolite2-asn.mmdb \
      https://cdn.jsdelivr.net/npm/@ip-location-db/geolite2-asn-mmdb/geolite2-asn.mmdb && \
    curl -fsSL -o licensed/SourceHanSansSC-Regular.otf \
      https://raw.githubusercontent.com/adobe-fonts/source-han-sans/refs/heads/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf && \
    curl -fsSL -o licensed/gsa_useragents.txt \
      https://raw.githubusercontent.com/searxng/searxng/refs/heads/master/searx/data/gsa_useragents.txt

RUN npm run build

RUN rm -rf ~/.config/google-chrome && \
    mkdir -p ~/.config/google-chrome && \
    NODE_COMPILE_CACHE=node_modules npm run dry-run

RUN if getent group 1000 >/dev/null; then \
        OLD_GROUP="$(getent group 1000 | cut -d: -f1)" && \
        groupmod -n tiger "$OLD_GROUP" || true; \
    else \
        groupadd -g 1000 tiger; \
    fi && \
    if id -u 1000 >/dev/null 2>&1; then \
        OLD_USER="$(getent passwd 1000 | cut -d: -f1)" && \
        usermod -l tiger -d /home/tiger -m -g 1000 -s /bin/sh "$OLD_USER" || true; \
    else \
        useradd -u 1000 -g 1000 -d /home/tiger -m -s /bin/sh tiger; \
    fi

RUN mkdir -p /home/tiger/.service \
    /workspace \
    /opt/tiger \
    /opt/log/tiger \
    /var/log/tiger && \
    chown -R tiger:tiger /home/tiger && \
    chown -R tiger:tiger /opt/tiger && \
    chown -R tiger:tiger /opt/log && \
    chown root:root /var/log/tiger && \
    chown tiger:root /workspace

RUN mkdir -p /etc/sudoers.d && \
    echo "tiger ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/tiger && \
    echo "tiger ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/sudoers && \
    chmod 0440 /etc/sudoers.d/tiger /etc/sudoers.d/sudoers

WORKDIR /workspace

EXPOSE 8080 8081

USER root

CMD ["/bin/bash"]
