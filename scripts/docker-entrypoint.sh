#!/bin/sh
set -e

# ==============================================================================
# Container Entrypoint: Database readiness & migration before service start
# ==============================================================================

echo "==> Competitive Intelligence Service Entrypoint"

# Wait for PostgreSQL to become responsive if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "==> Verifying database readiness..."
    python - <<'EOF'
import asyncio
import os
import sys
import time

db_url = os.environ.get("DATABASE_URL", "")
if "postgresql" in db_url:
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text

    async def wait_for_db():
        max_retries = 30
        engine = create_async_engine(db_url, pool_pre_ping=True)
        for attempt in range(1, max_retries + 1):
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                print(f"==> Database connected successfully on attempt {attempt}.")
                await engine.dispose()
                return 0
            except Exception as err:
                print(f"==> Waiting for database (attempt {attempt}/{max_retries}): {err}")
                await asyncio.sleep(2)
        print("==> ERROR: Database could not be reached after maximum retries.", file=sys.stderr)
        await engine.dispose()
        return 1

    sys.exit(asyncio.run(wait_for_db()))
else:
    print("==> Non-PostgreSQL database configured, skipping connection wait.")
EOF

    # Execute Alembic migrations
    echo "==> Applying database migrations (alembic upgrade head)..."
    python -m alembic upgrade head
    echo "==> Database migrations applied successfully."
fi

# Execute the container command
echo "==> Executing: $@"
exec "$@"
