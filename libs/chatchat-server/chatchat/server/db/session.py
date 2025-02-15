from contextlib import contextmanager
from functools import wraps

from sqlalchemy.orm import Session

from chatchat.server.db.base import SessionLocal


@contextmanager
def session_scope() -> Session:
    """上下文管理器用于自动获取 Session, 避免错误"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

# 当装饰的函数被调用时，with_session 首先创建一个数据库会话，并将这个会话作为第一个参数传递给被装饰的函数。
# 在被装饰的函数执行过程中，如果操作成功完成，则会话会被提交，以保存更改。
# 如果在执行过程中遇到任何异常，则会话会被回滚，以撤销所有未提交的更改，然后异常会被重新抛出。
# 这样，使用with_session装饰的函数可以专注于业务逻辑，而不必担心数据库会话的管理。
def with_session(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with session_scope() as session: # 与上面session_scope()联合使用
            try:
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise

    return wrapper


def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> SessionLocal:
    db = SessionLocal()
    return db
