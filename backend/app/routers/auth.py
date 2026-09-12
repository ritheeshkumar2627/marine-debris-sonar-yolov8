from fastapi import APIRouter

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
async def register():
    return {"message": "Register placeholder"}

@router.post("/login")
async def login():
    return {"message": "Login placeholder"}

@router.get("/users/me")
async def get_current_user():
    return {"message": "Get current user placeholder"}
