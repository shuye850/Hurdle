# src/model.py
from __future__ import annotations


BODY17_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


WHOLEBODY_FOOT_KEYPOINT_NAMES = [
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
]


def build_wholebody133_keypoint_names() -> list[str]:
    """
    构建 wholebody 133 点名称表。

    结构约定：
    - 0~16: body17
    - 17~22: foot6
    - 23~90: face68
    - 91~111: left_hand21
    - 112~132: right_hand21

    Returns:
        list[str]: 长度为 133 的关键点名称列表
    """
    names: list[str] = []

    # body 17
    names.extend(BODY17_KEYPOINT_NAMES)

    # foot 6
    names.extend(WHOLEBODY_FOOT_KEYPOINT_NAMES)

    # face 68
    for i in range(68):
        names.append(f"face_{i}")

    # left hand 21
    for i in range(21):
        names.append(f"left_hand_{i}")

    # right hand 21
    for i in range(21):
        names.append(f"right_hand_{i}")

    return names


WHOLEBODY133_KEYPOINT_NAMES = build_wholebody133_keypoint_names()


BODY17_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


WHOLEBODY_FOOT_SKELETON = [
    (15, 17),  # left_ankle -> left_big_toe
    (15, 18),  # left_ankle -> left_small_toe
    (15, 19),  # left_ankle -> left_heel
    (16, 20),  # right_ankle -> right_big_toe
    (16, 21),  # right_ankle -> right_small_toe
    (16, 22),  # right_ankle -> right_heel
]


BODY17_SCHEMA = {
    "schema_name": "body17",
    "num_keypoints": 17,
    "keypoint_names": BODY17_KEYPOINT_NAMES,
    "skeleton": BODY17_SKELETON,
    "pose2d_alias": "human",
    "description": "仅身体17点，不含脚部扩展点、手部、面部。",
}


WHOLEBODY133_SCHEMA = {
    "schema_name": "wholebody133",
    "num_keypoints": 133,
    "keypoint_names": WHOLEBODY133_KEYPOINT_NAMES,
    "skeleton": BODY17_SKELETON + WHOLEBODY_FOOT_SKELETON,
    "pose2d_alias": "wholebody",
    "description": "全点模式：身体17点 + 脚部6点 + 面部68点 + 双手42点。",
}


def get_schema(schema_name: str) -> dict:
    """
    根据名称获取关键点方案配置。

    Args:
        schema_name: 方案名，可选：
            - body17
            - wholebody133

    Returns:
        dict: 对应的 schema 配置字典

    Raises:
        ValueError: 当 schema_name 不合法时抛出
    """
    schema_map = {
        "body17": BODY17_SCHEMA,
        "wholebody133": WHOLEBODY133_SCHEMA,
    }

    if schema_name not in schema_map:
        raise ValueError(
            f"不支持的 schema_name: {schema_name}，"
            f"可选值为: {list(schema_map.keys())}"
        )

    return schema_map[schema_name]


def get_keypoint_names(schema_name: str) -> list[str]:
    """
    获取指定方案的关键点名称列表。

    Args:
        schema_name: 方案名

    Returns:
        list[str]: 关键点名称列表
    """
    return get_schema(schema_name)["keypoint_names"]


def get_num_keypoints(schema_name: str) -> int:
    """
    获取指定方案的关键点数量。

    Args:
        schema_name: 方案名

    Returns:
        int: 关键点数量
    """
    return int(get_schema(schema_name)["num_keypoints"])


def get_skeleton(schema_name: str) -> list[tuple[int, int]]:
    """
    获取指定方案的骨架连接定义。

    Args:
        schema_name: 方案名

    Returns:
        list[tuple[int, int]]: 骨架连接列表
    """
    return get_schema(schema_name)["skeleton"]


def get_default_pose2d_alias(schema_name: str) -> str:
    """
    获取指定方案建议使用的 pose2d 模型别名。

    Args:
        schema_name: 方案名

    Returns:
        str: pose2d 别名
    """
    return str(get_schema(schema_name)["pose2d_alias"])