from fastapi import FastAPI
from routers.rag import router as rag_router, item_not_found_handler
from exceptions.rag import ItemNotFoundError


app = FastAPI()

# Register exception handler on application (APIRouter doesn't provide this decorator)
app.add_exception_handler(ItemNotFoundError, item_not_found_handler)

app.include_router(rag_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)