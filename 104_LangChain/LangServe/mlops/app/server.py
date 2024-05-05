from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langserve import RemoteRunnable
from pirate_speak.chain import chain as pirate_speak_chain

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

add_routes(app, pirate_speak_chain, path="/pirate-speak")


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
#add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
    #api = RemoteRunnable("http://127.0.0.1:8000/pirate-speak")
    #api.invoke({"text": "hi"})
