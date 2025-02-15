import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

from chatchat.settings import Settings

# 数据库链接引擎，且将查询出的class使用json序列化
engine = create_engine(
    Settings.basic_settings.SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

# 创建一个DBSession，是与数据库链接的接口
# DBSession对象可视为当前数据库连接
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# orm表的基类
Base: DeclarativeMeta = declarative_base()
