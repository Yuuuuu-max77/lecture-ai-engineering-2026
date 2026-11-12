"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai.messages import DeferredToolRequests, DeferredToolResults, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, load_history, save_history

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_read_lecture_answer_is_registered_and_called():
    model = TestModel(call_tools=["read_lecture_answer"])
    with agent.override(model=model):
        result = agent.run_sync("第3回の解答ファイルを読んで")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert "read_lecture_answer" in offered
    assert "read_lecture_answer" in called


def test_make_quiz_is_registered_and_called():
    model = TestModel(call_tools=["make_quiz"])
    with agent.override(model=model):
        result = agent.run_sync("第2回の確認問題を出して")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert "make_quiz" in offered
    assert "make_quiz" in called


def test_save_learning_log_waits_for_approval(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("学習記録を保存して")

    offered = {t.name: t for t in model.last_model_request_parameters.function_tools}
    assert "save_learning_log" in offered
    assert isinstance(pending.output, DeferredToolRequests)
    assert len(pending.output.approvals) == 1
    assert not (tmp_path / "learning_log").exists()

    approval = pending.output.approvals[0]
    with agent.override(model=TestModel(call_tools=[])):
        completed = agent.run_sync(
            "学習記録を保存して",
            message_history=pending.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={approval.tool_call_id: True}
            ),
        )

    assert "保存しました" in completed.output
    assert (tmp_path / "learning_log" / "learning_log.md").exists()


def test_save_learning_log_is_not_written_when_denied(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    with agent.override(model=TestModel(call_tools=["save_learning_log"])):
        pending = agent.run_sync("学習記録を保存して")

    assert isinstance(pending.output, DeferredToolRequests)
    approval = pending.output.approvals[0]
    with agent.override(model=TestModel(call_tools=[])):
        agent.run_sync(
            "学習記録を保存して",
            message_history=pending.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={approval.tool_call_id: False}
            ),
        )

    assert not (tmp_path / "learning_log").exists()


def test_conversation_history_round_trips_between_runs(tmp_path):
    history_path = tmp_path / "conversation.json"
    first_model = TestModel(call_tools=[])
    with agent.override(model=first_model):
        first = agent.run_sync("最初の質問", model=first_model)
    save_history(first.all_messages(), history_path)

    history = load_history(history_path)
    second_model = TestModel(call_tools=[])
    with agent.override(model=second_model):
        second = agent.run_sync(
            "続きの質問",
            model=second_model,
            message_history=history,
        )
    save_history(second.all_messages(), history_path)

    assert len(second.all_messages()) > len(first.all_messages())
    assert "最初の質問" in history_path.read_text(encoding="utf-8")
    assert "続きの質問" in history_path.read_text(encoding="utf-8")


def test_load_history_returns_empty_for_missing_file(tmp_path):
    assert load_history(tmp_path / "missing.json") == []
