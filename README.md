# Building a Docker Image and Creating a Container

## Step 1: Build the Docker Image

To build the Docker image, run the following command in your terminal:

```bash
docker build -t streamlit-app .
```

## Step 2: Create and Run a Container

After building the image, you can create and run a container using the following command:

```bash
docker run -d \
    --name ajsa-app \
    -p 11434:11434 \
    -e PORT=11434 \
    --restart unless-stopped \
    -v $(pwd)/ajsa_database.sqlite3:/app/ajsa_database.sqlite3 \
    -v $(pwd)/.env:/app/.env \
    streamlit-app
```

**⚠️ Warning:** Ensure that the `ajsa_database.sqlite3` and `.env` files exist in the current directory before running the container.

## Step 3: Verify the Container is Running

If you want to verify everything is working:

```bash
# Check container status
docker ps

# View container logs
docker logs ajsa-app

# Check the volume mount
docker inspect ajsa-app | grep -A 10 Mounts
```
