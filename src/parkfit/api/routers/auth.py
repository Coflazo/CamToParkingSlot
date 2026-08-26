"""Registration, login and account preferences."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from parkfit.api.schemas import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserPreferencesUpdate,
    UserResponse,
)
from parkfit.api.security import (
    create_access_token,
    get_current_user,
    hash_password,
    needs_rehash,
    verify_password,
)
from parkfit.storage.models import User
from parkfit.storage.session import get_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest, session: AsyncSession = Depends(get_session)
) -> TokenResponse:
    email = payload.email.lower()
    existing = (
        await session.execute(select(User).where(func.lower(User.email) == email))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="That email is already registered"
        )

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        display_name=payload.display_name,
    )
    session.add(user)
    await session.flush()

    token, expires_in = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_session)) -> TokenResponse:
    email = payload.email.lower()
    user = (
        await session.execute(select(User).where(func.lower(User.email) == email))
    ).scalar_one_or_none()

    # Verify against a dummy hash when the account does not exist, so the response time
    # does not reveal which emails are registered.
    stored = user.password_hash if user else _DUMMY_HASH
    if not verify_password(payload.password, stored) or user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)

    token, expires_in = create_access_token(user.id, user.email)
    return TokenResponse(access_token=token, expires_in=expires_in)


@router.get("/me", response_model=UserResponse)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user)


@router.patch("/me", response_model=UserResponse)
async def update_preferences(
    payload: UserPreferencesUpdate,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> UserResponse:
    for field, value in payload.model_dump(exclude_unset=True).items():
        if value is not None:
            setattr(user, field, value)
    await session.flush()
    return UserResponse.model_validate(user)


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    user: User = Depends(get_current_user), session: AsyncSession = Depends(get_session)
) -> None:
    """Delete the account and everything attached to it.

    Vehicles cascade. This is a genuine deletion rather than a flag, because the data
    involved -- vehicle dimensions and, if enabled, destination history -- is personal
    data and a right to erasure is not satisfied by hiding a row.
    """
    await session.delete(user)
    await session.flush()


#: A real Argon2 hash of a value nobody will guess, used only to equalise login timing.
_DUMMY_HASH = hash_password("parkfit-timing-equaliser-not-a-real-password")
