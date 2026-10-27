"""工具定义与执行。

工具是 Agent 区别于普通问答的关键：模型决定"要不要查、查什么"，
执行结果回灌给模型后再组织答案，因此能回答知识库里没有的实时数据问题。
"""
import json

# OpenAI Function Calling Schema
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_device_status",
            "description": "查询指定设备的实时运行状态，返回状态、功率、温度、最近更新时间。设备编号示例：INV-01 逆变器、ESS-01 储能舱、CB-03 汇流箱。",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "设备编号，如 INV-01"}
                },
                "required": ["device_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_generation",
            "description": "查询某光伏场站指定日期的发电量数据，返回日发电量、等效利用小时数、辐照量。",
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {"type": "string", "description": "场站编号，如 PV-JH-01"},
                    "date": {"type": "string", "description": "日期，格式 YYYY-MM-DD"},
                },
                "required": ["station_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_alarms",
            "description": "查询场站当前告警列表，可按等级过滤（critical/major/minor/all）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "station_id": {"type": "string", "description": "场站编号"},
                    "level": {"type": "string", "description": "告警等级：critical/major/minor/all"},
                },
                "required": ["station_id"],
            },
        },
    },
]


# ---------------- 模拟的业务数据源（真实项目里换成 SCADA / 数据库 / 接口） ----------------
def _hash_salt(s):
    return sum(ord(c) for c in s)


def query_device_status(device_id: str):
    base = _hash_salt(device_id)
    states = ["运行", "运行", "运行", "降载运行", "停机", "故障"]
    state = states[base % len(states)]
    return json.dumps({
        "device_id": device_id,
        "status": state,
        "active_power_kw": round(400 + base % 260, 1) if state != "停机" else 0,
        "temperature_c": round(35 + base % 15, 1),
        "updated_at": "2026-10-27 14:00:00",
    }, ensure_ascii=False)


def query_generation(station_id: str, date: str):
    base = _hash_salt(station_id + date)
    kwh = 82000 + base % 26000
    return json.dumps({
        "station_id": station_id,
        "date": date,
        "daily_generation_kwh": kwh,
        "equivalent_hours": round(kwh / 50000, 2),
        "radiation_mj_m2": round(12 + base % 9, 2),
    }, ensure_ascii=False)


def query_alarms(station_id: str, level: str = "all"):
    pool = [
        {"id": "ALM-2311", "level": "critical", "device": "INV-02",
         "text": "逆变器绝缘阻抗低（PV-ISO），已闭锁合闸", "time": "13:42"},
        {"id": "ALM-2312", "level": "major", "device": "STR-05",
         "text": "组串电流为零，疑似熔断器熔断", "time": "13:20"},
        {"id": "ALM-2313", "level": "minor", "device": "CB-03",
         "text": "汇流箱防雷器状态窗变红，待更换", "time": "11:05"},
        {"id": "ALM-2314", "level": "minor", "device": "ESS-01",
         "text": "电池舱内单体温差 6 摄氏度，超过 5 摄氏度阈值", "time": "09:38"},
    ]
    data = pool if level == "all" else [a for a in pool if a["level"] == level]
    return json.dumps({"station_id": station_id, "count": len(data), "alarms": data},
                      ensure_ascii=False)


REGISTRY = {
    "query_device_status": query_device_status,
    "query_generation": query_generation,
    "query_alarms": query_alarms,
}


def execute(name, arguments):
    fn = REGISTRY.get(name)
    if not fn:
        return json.dumps({"error": f"未注册的工具：{name}"}, ensure_ascii=False)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        args = {}
    try:
        return fn(**args)
    except TypeError as e:
        # 参数不合法时把错误回传给模型，让它自己纠正 —— 而不是直接抛异常
        return json.dumps({"error": f"参数错误：{e}"}, ensure_ascii=False)
