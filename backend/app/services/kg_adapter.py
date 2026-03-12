"""
知识图谱适配器
支持 Zep Cloud 和 Graphiti (本地) 两种模式

使用方式:
    from app.services.kg_adapter import get_knowledge_graph_adapter

    kg = get_knowledge_graph_adapter()
    kg.add_episode(graph_id="xxx", text="hello")
    kg.search(graph_id="xxx", query="hello")
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

from ..config import Config

logger = logging.getLogger(__name__)


class KnowledgeGraphAdapter(ABC):
    """知识图谱适配器抽象基类"""

    @abstractmethod
    def create_graph(self, graph_id: str, name: str = None) -> Any:
        """创建图谱"""
        pass

    @abstractmethod
    def add_episode(self, graph_id: str, text: str, **kwargs) -> Any:
        """添加单条内容"""
        pass

    @abstractmethod
    def add_episodes_batch(self, graph_id: str, texts: List[str]) -> List[Any]:
        """批量添加内容"""
        pass

    @abstractmethod
    def get_episode(self, episode_uuid: str) -> Any:
        """获取单个 episode"""
        pass

    @abstractmethod
    def search(self, graph_id: str, query: str, limit: int = 10) -> List[Dict]:
        """搜索"""
        pass

    @abstractmethod
    def get_nodes(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        """获取节点"""
        pass

    @abstractmethod
    def get_node(self, node_uuid: str) -> Any:
        """获取单个节点"""
        pass

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[Dict]:
        """获取节点的所有边"""
        pass

    @abstractmethod
    def get_edges(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        """获取边"""
        pass

    @abstractmethod
    def delete(self, graph_id: str) -> bool:
        """删除图谱"""
        pass

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: Dict) -> bool:
        """设置本体"""
        pass

    @abstractmethod
    def get_graph_info(self, graph_id: str) -> Dict:
        """获取图谱信息"""
        pass


class ZepCloudAdapter(KnowledgeGraphAdapter):
    """Zep Cloud 适配器"""

    def __init__(self, api_key: str = None):
        from zep_cloud.client import Zep
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY 未配置")
        self.client = Zep(api_key=self.api_key)
        logger.info("ZepCloudAdapter 初始化完成")

    def create_graph(self, graph_id: str, name: str = None) -> Any:
        return self.client.graph.create(graph_id=graph_id, name=name or graph_id)

    def add_episode(self, graph_id: str, text: str, **kwargs) -> Any:
        return self.client.graph.add(graph_id=graph_id, type="text", data=text)

    def add_episodes_batch(self, graph_id: str, texts: List[str]) -> List[Any]:
        from zep_cloud.types import EpisodeData
        episodes = [EpisodeData(data=t, type="text") for t in texts]
        return self.client.graph.add_batch(episodes=episodes, graph_id=graph_id)

    def get_episode(self, episode_uuid: str) -> Any:
        return self.client.graph.episode.get(uuid_=episode_uuid)

    def search(self, graph_id: str, query: str, limit: int = 10) -> List[Dict]:
        result = self.client.graph.search(graph_id=graph_id, query=query, limit=limit)
        return [r.model_dump() for r in result.results] if hasattr(result, 'results') else []

    def get_nodes(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        kwargs = {"limit": limit}
        if cursor:
            kwargs["uuid_cursor"] = cursor
        return self.client.graph.node.get_by_graph_id(graph_id=graph_id, **kwargs)

    def get_node(self, node_uuid: str) -> Any:
        return self.client.graph.node.get(uuid_=node_uuid)

    def get_node_edges(self, node_uuid: str) -> List[Dict]:
        edges = self.client.graph.node.get_entity_edges(node_uuid=node_uuid)
        return [e.model_dump() if hasattr(e, 'model_dump') else e for e in edges]

    def get_edges(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        kwargs = {"limit": limit}
        if cursor:
            kwargs["uuid_cursor"] = cursor
        return self.client.graph.edge.get_by_graph_id(graph_id=graph_id, **kwargs)

    def delete(self, graph_id: str) -> bool:
        self.client.graph.delete(graph_id=graph_id)
        return True

    def set_ontology(self, graph_id: str, ontology: Dict) -> bool:
        entities = ontology.get('entities', {})
        edges = ontology.get('edges', {})
        self.client.graph.set_ontology(
            entities=entities,
            edges=edges,
            graph_ids=[graph_id]
        )
        return True

    def get_graph_info(self, graph_id: str) -> Dict:
        # Zep Cloud 没有直接的图谱信息 API，返回基本信息
        return {"graph_id": graph_id}


class GraphitiAdapter(KnowledgeGraphAdapter):
    """Graphiti 适配器 - 本地部署"""

    def __init__(self):
        import os
        import asyncio
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_client import OpenAIClient
        from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

        if not all([Config.NEO4J_URI, Config.NEO4J_USER, Config.NEO4J_PASSWORD]):
            raise ValueError("Neo4j 配置不完整，请检查 NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")

        # 获取 API Key（优先使用独立配置，其次使用 LLM 配置）
        api_key = Config.LLM_API_KEY or Config.OPENAI_API_KEY
        llm_base_url = Config.LLM_BASE_URL

        # 嵌入模型独立配置
        embedding_api_key = Config.EMBEDDING_API_KEY or api_key
        embedding_base_url = Config.EMBEDDING_BASE_URL or llm_base_url

        if not api_key:
            raise ValueError("请配置 LLM_API_KEY")

        # 设置环境变量（Graphiti 内部组件会读取）
        os.environ['OPENAI_API_KEY'] = api_key
        os.environ['OPENAI_BASE_URL'] = llm_base_url

        # 配置 LLM 客户端（支持 OpenAI 兼容 API）
        llm_config = LLMConfig(
            api_key=api_key,
            base_url=llm_base_url,
            model=Config.LLM_MODEL_NAME,
            small_model=Config.LLM_MODEL_NAME,  # 使用相同模型
        )
        llm_client = OpenAIClient(config=llm_config)

        # 配置 Embedder 客户端（可独立配置）
        embedder_config = OpenAIEmbedderConfig(
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            embedding_model=Config.EMBEDDING_MODEL,
            embedding_dim=Config.EMBEDDING_DIM,
        )
        embedder_client = OpenAIEmbedder(config=embedder_config)

        self.client = Graphiti(
            uri=Config.NEO4J_URI,
            user=Config.NEO4J_USER,
            password=Config.NEO4J_PASSWORD,
            llm_client=llm_client,
            embedder=embedder_client,
            cross_encoder=None,  # 禁用 reranker，需要时可配置
        )
        # graph_id 到 group 的映射（Graphiti 使用 group 区分不同的图）
        self._graph_id_to_group: Dict[str, str] = {}

        # 使用同步驱动避免 asyncio 事件循环冲突
        from neo4j import GraphDatabase
        self._sync_driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )

        # 初始化数据库索引
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.client.build_indices_and_constraints())
            loop.close()
            logger.info("Graphiti 数据库索引初始化完成")
        except Exception as e:
            logger.warning(f"数据库索引初始化警告: {e}")

        logger.info("GraphitiAdapter 初始化完成")

    def _run_async(self, coro):
        """同步调用异步方法的包装器，使用持久化事件循环"""
        import asyncio

        # 创建持久化的事件循环（线程级别）
        if not hasattr(self, '_loop') or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

        result = self._loop.run_until_complete(coro)
        return result

    def _get_group(self, graph_id: str) -> str:
        """获取或创建 group"""
        if graph_id not in self._graph_id_to_group:
            self._graph_id_to_group[graph_id] = graph_id
        return self._graph_id_to_group[graph_id]

    def create_graph(self, graph_id: str, name: str = None) -> Any:
        # Graphiti 不需要预创建图，通过 group 区分
        self._graph_id_to_group[graph_id] = graph_id

        # 创建 Group 节点
        with self._sync_driver.session() as session:
            session.run("""
                MERGE (g:Group {name: $name})
                SET g.created_at = datetime()
            """, name=graph_id)

        logger.info(f"Graphiti: 标记图谱 {graph_id}")
        return {"status": "ok", "graph_id": graph_id}

    def add_episode(self, graph_id: str, text: str, **kwargs) -> Any:
        """使用同步驱动添加 episode"""
        import uuid
        from datetime import datetime, timezone

        group = self._get_group(graph_id)
        episode_uuid = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # 直接使用同步驱动创建 episode
        with self._sync_driver.session() as session:
            query = """
            CREATE (e:Episodic {
                uuid: $uuid,
                name: $name,
                content: $content,
                created_at: $created_at,
                valid_at: $valid_at,
                group_id: $group_id,
                source: 'text',
                episode_type: 'text'
            })
            RETURN e
            """
            result = session.run(
                query,
                uuid=episode_uuid,
                name=f"episode_{now.strftime('%Y%m%d%H%M%S')}",
                content=text,
                created_at=now,
                valid_at=now,
                group_id=group
            )
            record = result.single()
            return {"uuid": episode_uuid, "name": record["e"]["name"]} if record else None

    def add_episodes_batch(self, graph_id: str, texts: List[str], batch_size: int = 10) -> List[Any]:
        """批量添加内容，使用同步驱动，并提取实体"""
        results = []
        group = self._get_group(graph_id)

        # 直接添加 episodes
        for i, text in enumerate(texts):
            result = self.add_episode(graph_id, text)
            results.append(result)
            if (i + 1) % batch_size == 0:
                logger.info(f"已添加 {i + 1}/{len(texts)} 条内容")

        # 提取实体
        self._extract_entities_from_texts(graph_id, texts)

        logger.info(f"图谱实体全部构建完成: {graph_id}, 共 {len(results)} 条")
        return results

    def _extract_entities_from_texts(self, graph_id: str, texts: List[str]):
        """从文本中提取实体并存储到 Neo4j"""
        import uuid

        # 合并所有文本进行实体提取
        combined_text = "\n\n".join(texts)

        # 使用 LLM 提取实体
        entities_json = self._call_llm_for_entities(combined_text)

        if not entities_json:
            logger.warning("未能从文本中提取到实体")
            return

        # 解析实体
        try:
            import json
            import re

            # 尝试直接解析
            try:
                entities_data = json.loads(entities_json)
            except json.JSONDecodeError:
                # 尝试提取 JSON 数组
                match = re.search(r'\[.*\]', entities_json, re.DOTALL)
                if match:
                    entities_data = json.loads(match.group())
                else:
                    raise ValueError("No JSON array found")

            if not entities_data or not isinstance(entities_data, list):
                logger.warning("未提取到实体数据")
                return

            logger.info(f"提取到 {len(entities_data)} 个实体")
        except Exception as e:
            logger.error(f"解析实体数据失败: {e}, 内容: {entities_json[:300]}")
            return

        # 存储实体到 Neo4j
        with self._sync_driver.session() as session:
            for entity in entities_data:
                entity_name = entity.get("name")
                entity_type = entity.get("type", "Entity")
                description = entity.get("description", "")
                relationships = entity.get("relationships", [])

                if not entity_name:
                    continue

                # 确保 entity_type 有效，否则使用默认标签
                if not entity_type or not entity_type.strip():
                    entity_label = "Entity"
                else:
                    entity_label = entity_type.strip()
                    # Neo4j 标签不能以数字开头
                    if entity_label[0].isdigit():
                        entity_label = f"Type{entity_label}"

                # 创建实体节点并关联到 Group
                # 实体同时拥有动态标签和 Entity 基类标签，便于查询
                entity_uuid = str(uuid.uuid4())
                query = f"""
                MERGE (g:Group {{name: $group_id}})
                MERGE (e:`{entity_label}` {{name: $name, group_id: $group_id}})
                SET e:Entity,
                    e.uuid = $uuid,
                    e.summary = $summary,
                    e.created_at = datetime(),
                    e.entity_type = $type
                MERGE (e)-[:MEMBER_OF]->(g)
                RETURN e
                """
                session.run(
                    query,
                    name=entity_name,
                    uuid=entity_uuid,
                    summary=description,
                    type=entity_type,
                    group_id=graph_id
                )

                # 创建关系
                for rel in relationships:
                    target = rel.get("target")
                    rel_type = rel.get("type", "RELATED_TO")
                    fact = rel.get("fact", "")

                    if target:
                        rel_query = """
                        MATCH (e1:Entity {name: $source, group_id: $group_id})
                        MATCH (e2:Entity {name: $target, group_id: $group_id})
                        MERGE (e1)-[r:RELATED {fact: $fact, fact_type: $type}]->(e2)
                        SET r.created_at = datetime()
                        """
                        session.run(
                            rel_query,
                            source=entity_name,
                            target=target,
                            group_id=graph_id,
                            fact=fact,
                            type=rel_type
                        )

        logger.info(f"实体提取完成: {len(entities_data)} 个实体")

    def _call_llm_for_entities(self, text: str) -> str:
        """调用 LLM 提取实体"""
        from openai import OpenAI

        client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL
        )

        prompt = f"""从以下文本中提取实体和关系。

重要：直接返回JSON数组，不要任何markdown格式，不要```标记。

要求返回格式：
[
  {{
    "name": "实体名",
    "type": "实体类型",
    "description": "描述",
    "relationships": [
      {{"target": "目标实体", "type": "关系类型", "fact": "事实描述"}}
    ]
  }}
]

文本内容：
{text[:3000]}

直接返回JSON数组："""

        try:
            response = client.chat.completions.create(
                model=Config.LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个实体关系提取助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            # 清理 JSON（去除 markdown 代码块）
            content = response.choices[0].message.content
            content = content.strip()
            # 去除 ```json 和 ``` 标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            return content.strip()
        except Exception as e:
            logger.error(f"LLM 实体提取失败: {e}")
            return "[]"

    def get_episode(self, episode_uuid: str) -> Any:
        """使用同步驱动获取 episode"""
        with self._sync_driver.session() as session:
            query = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.content as content, e.created_at as created_at,
                   e.valid_at as valid_at, e.uuid as uuid,
                   e.name as name, e.group_id as group_id
            """
            result = session.run(query, uuid=episode_uuid)
            record = result.single()
            if record:
                return dict(record)
            return None

    def search(self, graph_id: str, query: str, limit: int = 10) -> List[Dict]:
        """使用同步驱动搜索（简单实现：搜索 episodes 内容）"""
        group = self._get_group(graph_id)
        with self._sync_driver.session() as session:
            # 简单的文本搜索：匹配 episodes 内容
            query_cypher = """
            MATCH (e:Episodic {group_id: $group})
            WHERE e.content CONTAINS $search_text
            RETURN e.content as content, e.uuid as uuid, e.name as name
            LIMIT $limit
            """
            result = session.run(
                query_cypher,
                group=group,
                search_text=query,
                limit=limit
            )
            return [{"content": r["content"], "uuid": r["uuid"], "name": r["name"]} for r in result]

    def get_nodes(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        """通过同步驱动查询实体节点"""
        with self._sync_driver.session() as session:
            query = """
            MATCH (e:Entity)-[:MEMBER_OF]->(g:Group {name: $group})
            RETURN e.uuid as uuid_, e.name as name, labels(e) as labels,
                   e.summary as summary, e.created_at as created_at,
                   e.entity_type as entity_type
            LIMIT $limit
            """
            result = session.run(query, group=graph_id, limit=limit)
            nodes = [dict(record) for record in result]
            # 转换格式以兼容前端，将 entity_type 放入 attributes
            for node in nodes:
                if 'attributes' not in node:
                    node['attributes'] = {}
                if node.get('entity_type'):
                    node['attributes']['entity_type'] = node['entity_type']
            return nodes

    def get_node(self, node_uuid: str) -> Any:
        """通过同步驱动获取单个节点"""
        with self._sync_driver.session() as session:
            query = """
            MATCH (e:Entity {uuid: $uuid})
            RETURN e.uuid as uuid_, e.name as name, labels(e) as labels,
                   e.summary as summary, e.created_at as created_at
            """
            result = session.run(query, uuid=node_uuid)
            record = result.single()
            if record:
                node = dict(record)
                if 'attributes' not in node:
                    node['attributes'] = {}
                return node
            return None

    def get_node_edges(self, node_uuid: str) -> List[Dict]:
        """通过同步驱动获取节点的所有边"""
        with self._sync_driver.session() as session:
            query = """
            MATCH (e1:Entity {uuid: $uuid})-[r]-(e2:Entity)
            RETURN r.uuid as uuid_, type(r) as name, r.fact as fact,
                   r.fact_type as fact_type,
                   e1.uuid as source_node_uuid, e2.uuid as target_node_uuid,
                   e1.name as source_node_name, e2.name as target_node_name,
                   r.created_at as created_at
            """
            result = session.run(query, uuid=node_uuid)
            edges = [dict(record) for record in result]
            return edges

    def get_edges(self, graph_id: str, limit: int = 100, cursor: str = None) -> List[Any]:
        """通过同步驱动查询边"""
        with self._sync_driver.session() as session:
            query = """
            MATCH (e1:Entity)-[r]-(e2:Entity)
            WHERE e1.group_id = $group OR e2.group_id = $group
            RETURN r.uuid as uuid_, type(r) as name, r.fact as fact,
                   r.fact_type as fact_type,
                   e1.uuid as source_node_uuid, e2.uuid as target_node_uuid,
                   e1.name as source_node_name, e2.name as target_node_name,
                   r.created_at as created_at, r.valid_at as valid_at,
                   r.invalid_at as invalid_at, r.expired_at as expired_at
            LIMIT $limit
            """
            result = session.run(query, group=graph_id, limit=limit)
            edges = [dict(record) for record in result]
            # 兼容前端格式
            for edge in edges:
                if 'attributes' not in edge:
                    edge['attributes'] = {}
                if 'episodes' not in edge:
                    edge['episodes'] = []
            return edges

    def delete(self, graph_id: str) -> bool:
        """使用同步驱动删除图谱"""
        with self._sync_driver.session() as session:
            # 删除关联边
            session.run("""
                MATCH (e1:Entity)-[r]-(e2:Entity)
                WHERE e1.group = $group OR e2.group = $group
                DELETE r
            """, group=graph_id)
            # 删除实体节点
            session.run("""
                MATCH (e:Entity)-[:MEMBER_OF]->(g:Group {name: $group})
                DELETE e
            """, group=graph_id)
            # 删除组节点
            session.run("""
                MATCH (g:Group {name: $group})
                DELETE g
            """, group=graph_id)

        if graph_id in self._graph_id_to_group:
            del self._graph_id_to_group[graph_id]

        logger.info(f"Graphiti: 删除图谱 {graph_id}")
        return True

    def set_ontology(self, graph_id: str, ontology: Dict) -> bool:
        # Graphiti 通过 Pydantic 模型定义实体类型
        # 这里简化处理：仅记录 ontology 配置
        logger.info(f"Graphiti: 设置本体 {graph_id}, ontology types: {list(ontology.keys())}")
        # 实际使用时需要动态创建实体类
        return True

    def get_graph_info(self, graph_id: str) -> Dict:
        """使用同步驱动获取图谱信息"""
        with self._sync_driver.session() as session:
            # 统计节点数量
            node_result = session.run("""
                MATCH (e:Entity)-[:MEMBER_OF]->(g:Group {name: $group})
                RETURN count(e) as count
            """, group=graph_id)
            node_count = node_result.single()["count"] if node_result.single() else 0

            # 统计边数量
            edge_result = session.run("""
                MATCH (e1:Entity)-[r]-(e2:Entity)
                WHERE e1.group = $group OR e2.group = $group
                RETURN count(r) as count
            """, group=graph_id)
            edge_count = edge_result.single()["count"] if edge_result.single() else 0

        return {
            "graph_id": graph_id,
            "node_count": node_count,
            "edge_count": edge_count,
        }

    def _result_to_dict(self, result) -> Dict:
        if hasattr(result, 'model_dump'):
            return result.model_dump()
        elif hasattr(result, 'dict'):
            return result.dict()
        return {}


# 全局缓存
_adapter_cache: Optional[KnowledgeGraphAdapter] = None


def get_knowledge_graph_adapter(force_refresh: bool = False) -> KnowledgeGraphAdapter:
    """
    获取知识图谱适配器实例

    Args:
        force_refresh: 是否强制刷新缓存

    Returns:
        KnowledgeGraphAdapter: 适配器实例
    """
    global _adapter_cache

    if _adapter_cache is not None and not force_refresh:
        return _adapter_cache

    mode = Config.KNOWLEDGE_GRAPH_MODE

    if mode == 'local':
        _adapter_cache = GraphitiAdapter()
    elif mode == 'cloud':
        _adapter_cache = ZepCloudAdapter()
    else:
        raise ValueError(f"未知的 KNOWLEDGE_GRAPH_MODE: {mode}")

    return _adapter_cache


def reset_adapter():
    """重置适配器缓存"""
    global _adapter_cache
    _adapter_cache = None
