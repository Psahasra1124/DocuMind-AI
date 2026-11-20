from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings


def get_storage_root() -> Path:
    return settings.DOCUMENT_STORAGE_DIR.resolve()


def generate_storage_path(
    owner_id: UUID,
    filename: str,
) -> Path:
    suffix = Path(filename).suffix.lower()
    return Path(str(owner_id)) / f"{uuid4()}{suffix}"


async def store_document(
    file: UploadFile,
    owner_id: UUID,
) -> str:
    storage_root = get_storage_root()
    relative_path = generate_storage_path(
        owner_id,
        file.filename or "",
    )
    destination = storage_root / relative_path

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        total_bytes = 0
        maximum_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        with destination.open("xb") as output_file:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > maximum_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=(
                            f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB} MB upload limit"
                        ),
                    )
                output_file.write(chunk)
    except FileExistsError:
        raise
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    return str(relative_path)


def delete_document(storage_path: str) -> None:
    storage_root = get_storage_root()

    destination = (
        storage_root / storage_path
    ).resolve()

    try:
        destination.relative_to(storage_root)
    except ValueError:
        raise ValueError(
            "Storage path must be inside storage root"
        )

    try:
        destination.unlink()
    except FileNotFoundError:
        pass