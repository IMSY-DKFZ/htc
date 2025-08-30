# Components which are needed for testing
FROM imsy-htc-base

# We need the Docker binaries for challenger-pydocker
RUN curl -fsSL https://get.docker.com -o /get-docker.sh \
 && sh /get-docker.sh \
 && rm /get-docker.sh

# Install the Libertinus font
RUN apt install -y fontconfig wget zstd \
 && mkdir -p /usr/share/fonts/opentype \
 && wget -O /usr/share/fonts/opentype/libertinus.tar.zst https://github.com/alerque/libertinus/releases/download/v7.051/Libertinus-7.051.tar.zst \
 && tar -xvf /usr/share/fonts/opentype/libertinus.tar.zst --directory /usr/share/fonts/opentype \
 && mv /usr/share/fonts/opentype/Libertinus-7.051/static/OTF /usr/share/fonts/opentype/libertinus \
 && rm -d -R -f /usr/share/fonts/opentype/Libertinus-7.051 \
 && rm /usr/share/fonts/opentype/libertinus.tar.zst \
 && fc-cache -f -v

# ImageMagick for PDF image comparison (used in paper tests)
RUN apt install -y libmagickwand-dev \
 && pip install Wand \
 && sed -i '/disable ghostscript format types/,+6d' /etc/ImageMagick-6/policy.xml

# Install chromium for kaleido, as it is explicitly required now
RUN apt update && apt install -y chromium-browser

# Install and cache the static code checks
# We need a git repo for pre-commit to work but we don't want to copy the main repo to the Docker container as it unnecessarily slows down the build process
# Instead, we create a new empty git repo and make sure that files are staged before the hooks run (because pre-commit does not consider unstaged files)
COPY .pre-commit-config.yaml /home/src/.pre-commit-config.yaml
RUN git init \
 && pre-commit install --install-hooks \
 && echo "git add . && pre-commit run --all-files --show-diff-on-failure" >> run_hooks.sh

# The code changes basically every time when we use the Docker container so we want to install it at the very end to make use of caching
COPY . /home/src

# We don't need an isolated build environment inside the Docker container and we already have the dependencies installed
# Further, it would significantly increase the startup time for the Docker container since torch is a build requirement and can take a long time to install (plus, it is a good idea to match the torch installation version with the torch version in the target environment)
RUN pip install --no-build-isolation -e /home/src

ENTRYPOINT ["python", "dependencies/docker_startup.py"]
