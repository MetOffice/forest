# Remote procedure call server

A simple FastAPI example to show
how remote datasets can be accessed with FOREST


```sh
# Run FastAPI server using uvicorn
uvicorn server:app
```

## 2022 edition FOREST config

At the time of writing the RPC (remote procedure call) driver is in
its infancy, but nonetheless can display/navigate image data.

```yaml
# forest.config.yaml
edition: 2022
datasets:
   - label: RPC example
     driver:
       name: rpc
       settings:
          url: 'http://localhost:8000'
```

By offloading data responsibilities to a server, while still tightly coupled to
procedures inside FOREST, the RPC driver enables datasets from multiple institutions
to be shared via the web.

## Running multiple servers

The downside to having datasets served remotely is multiple servers/processes need maintenance.

- Start the RPC server(s) on port 8000 with `./server.py`
- Start the forest server using `forest ctl forest.config.yaml` by default it will serve on port `5006`
