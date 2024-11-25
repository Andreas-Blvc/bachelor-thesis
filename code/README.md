### Jupytur 

Run in docker container:

```bash
python -c "import secrets; print(secrets.token_hex(32))" > jupyter_token.txt
docker build -t my-jupyter-notebook .
docker run -d -p 8888:8888 my-jupyter-notebook
```

Sync project with docker container:

- run `docker ps` to get the container id
- `./sync_project_to_container_rsync.sh <container_id>`


