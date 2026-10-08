"""Data contract validation. Model weights cannot identify a user's intended dataset."""
import hashlib
import json
import copy
from .inspection import permitted

EXAMPLES = {
    "cpt": {"text": "有权用于继续预训练的完整文本。"},
    "sft": {"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "优质回答"}]},
    "dpo": {"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "优选回答"}], "rejected_response": "劣选回答"},
    "grpo": {"messages": [{"role": "user", "content": "计算 2+3"}], "solution": "5"},
    "ppo": {"messages": [{"role": "user", "content": "问题"}]},
}


def dataset_guide(task, model=None, language='en'):
    """Backend-specific contracts; naming a model does not certify its adapter."""
    task={'pretrain':'cpt','rlfh':'rlhf'}.get(task,task)
    if task not in (*EXAMPLES,'rlhf'):
        raise ValueError('Unknown training objective')
    model=model or {}
    modality=model.get('modality','unknown')
    sample=copy.deepcopy(EXAMPLES.get(task,EXAMPLES['dpo']))
    fields={
        'cpt':[('text','string','Raw document; next-token targets are constructed by the tokenizer/trainer.')],
        'sft':[('messages','list[role, content]','Conversation ending with a non-empty assistant target.')],
        'dpo':[('messages','list[role, content]','Shared prompt followed by the preferred assistant response.'),('rejected_response','string','Rejected answer to exactly the same prompt, not a second unrelated question.')],
        'ppo':[('messages','list[role, content]','Prompt ending in user; policy generates the answer online.')],
        'grpo':[('messages','list[role, content]','Prompt ending in user; generate multiple candidates per prompt.'),('solution','string','Ground truth for this workbench mathematical accuracy reward; not a universal GRPO field.')],
        'rlhf':[('stage','workflow','Preference collection → reward-model training → PPO, or a direct preference route such as DPO. Not a single JSONL schema.')],
    }
    flows={
        'cpt':'document → tokenize / EOS / packing → shifted next-token labels → language-model loss',
        'sft':'conversation → model chat template → tokenize → assistant loss mask → supervised loss',
        'dpo':'same prompt + chosen/rejected → policy/reference log probabilities → preference loss',
        'ppo':'prompt → policy rollout → reward + critic value → advantages → clipped policy/value updates',
        'grpo':'prompt → group of rollouts → reward per answer → group-relative advantages → policy update',
        'rlhf':'human preferences → reward model → online policy optimization; DPO is an alternative preference-training route',
    }
    notes=['JSONL: one complete JSON object per line; the formatted preview must be serialized to one line.',
           'Use the selected model tokenizer/processor and chat template. Do not manually concatenate family-specific control tokens.',
           'Schema validation is not a tokenizer/processor dry-run: check EOS, loss masks, truncation, media decoding and train/validation leakage in the backend.',
           'These contracts target the current ms-swift adapter, not every trainer API. The local execution preflight still decides support.']
    if task=='cpt': notes.append('This tool continues downloaded weights; it does not initialize a model from scratch. CPT uses the adapter pretraining mode, not SFT chat masking.')
    if task=='dpo': notes.append('Reference log probabilities may be precomputed; do not include them as text in the prompt.')
    if task in ('ppo','rlhf'): notes.append('PPO/reward-model orchestration is not executable in this preview. A schema example is not proof of an implemented trainer.')
    if task=='grpo': notes.append('The executable recipe is restricted to the configured mathematical reward. General preference or robot rewards need a separate verified reward adapter.')
    if modality=='vision-language' and task in ('sft','dpo'):
        sample['messages'][0]['content']='<image>Describe the object in this image.'
        sample['messages'][-1]['content']='A red cup.'
        sample['images']=['/absolute/path/to/cup.jpg']
        if task=='dpo':sample['rejected_response']='A blue car.'
        fields[task].append(('images','list[path]','Local image paths in placeholder order; files must exist on the training host.'))
        notes.append('Verify the selected processor image-token expansion and pixel budget; image resolution changes memory even when text length is unchanged.')
    elif modality not in ('text','unknown'):
        notes.append('This modality/task needs a dedicated processor contract; the text skeleton below is NOT a validated multimedia sample.')
    if language=='zh':
        meanings={'text':'原始文档正文；分词后由训练器生成右移一位的预测标签。','messages':'按顺序排列的对话。SFT/DPO以assistant答案结尾；PPO/GRPO只提供待回答的问题，以user结尾。','rejected_response':'同一个问题的劣选回答，与messages末尾的优选回答构成偏好对。','solution':'当前数学accuracy奖励使用的标准答案，不属于所有GRPO任务的通用字段。','stage':'RLHF是一套流程，需选择奖励模型训练、PPO或DPO等具体阶段。','images':'按<image>占位符顺序提供训练主机可访问的图片路径。'}
        fields[task]=[(n,k,meanings.get(n,v)) for n,k,v in fields[task]]
        flows={'cpt':'文档 → 分词/EOS/拼接 → 下一token标签 → 语言模型损失','sft':'对话 → 模型专属聊天模板 → 分词 → assistant损失掩码 → 监督训练','dpo':'同题优选/劣选答案 → 策略与参考模型对数概率 → 偏好损失','ppo':'问题 → 策略生成 → 奖励与价值估计 → 优势 → 策略/价值更新','grpo':'问题 → 同题多次生成 → 逐答案奖励 → 组内相对优势 → 策略更新','rlhf':'人类偏好 → 奖励模型 → 在线策略优化；也可走DPO直接偏好训练路线'}
        notes=['JSONL每行必须是一条完整JSON对象；下方提供排版预览和可保存的单行形式。',
               '使用选中模型的tokenizer/processor和聊天模板，不要手拼模型家族专用控制token。',
               '结构校验不等于实际数据可训练：还须在后端检查EOS、损失掩码、截断、媒体解码、训练验证集泄漏。',
               '格式针对本工具的ms-swift适配器；模型名称不等于已支持训练，执行前仍须通过模型与后端预检。']
        if task=='cpt':notes.append('当前是在已下载权重上继续预训练，不是随机初始化从零训练；需使用预训练模式而非SFT聊天掩码。')
        if task=='grpo':notes.append('目前可执行的是数学奖励配方，solution供奖励函数使用，不拼入问题；其他奖励需另行实现并验证。')
        if task in ('ppo','rlhf'):notes.append('目前尚未接入PPO/奖励模型多角色执行器；此处仅提供数据准备说明，不能据此直接启动完整流程。')
        if modality=='vision-language' and task in ('sft','dpo'):notes.append('图片分辨率会改变视觉token数量和显存；必须使用该模型的processor验证，不能只计算文字长度。')
        elif modality not in ('text','unknown'):notes.append('此模态/任务尚无专用数据适配器；下面的文本骨架不是已验证的音视频训练样例。')
    return {'task':task,'model':model.get('name','Not selected'),'architecture':model.get('model_type','unknown'),
            'modality':modality,'fields':[dict(name=n,type=k,meaning=v) for n,k,v in fields[task]],
            'flow':flows[task],'sample':sample,'jsonl':json.dumps(sample,ensure_ascii=False),
            'notes':notes,'execution_status':'workflow-only' if task=='rlhf' else 'blocked' if task=='ppo' else 'requires-preflight',
            'source':'https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html'}


def validate_dataset(path, task, roots):
    if task not in EXAMPLES:
        raise ValueError("未知数据任务")
    p = permitted(path, roots)
    if p.suffix != ".jsonl":
        raise ValueError("首版要求 JSONL；先显式转换，避免静默猜列")
    errors, rows, digest, formats = [], 0, hashlib.sha256(), set()
    with p.open("rb") as f:
        for i, raw in enumerate(f, 1):
            digest.update(raw)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("单行超过 8 MiB；媒体使用文件引用，不内嵌巨型 base64")
            if not raw.strip():
                continue
            rows += 1
            try:
                r = json.loads(raw)
                if not isinstance(r, dict):
                    raise ValueError("必须是对象")
                if task == "cpt":
                    if not isinstance(r.get("text"), str) or not r["text"].strip():
                        raise ValueError("CPT 需要非空 text")
                    formats.add("text")
                else:
                    msgs = r.get("messages")
                    if not isinstance(msgs, list) or not msgs:
                        raise ValueError("需要非空 messages")
                    for m in msgs:
                        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool", "tool_call", "tool_response"):
                            raise ValueError("message role 非法")
                        if not isinstance(m.get("content"), str):
                            raise ValueError("当前规范要求 content 为文本，媒体放 images/videos/audios")
                    if task in ("sft", "dpo") and msgs[-1]["role"] != "assistant":
                        raise ValueError("监督样本最后必须是 assistant")
                    if task in ("grpo", "ppo") and msgs[-1]["role"] != "user":
                        raise ValueError("在线 RL 样本应结束于 user，避免答案泄漏")
                    if task == "dpo" and not isinstance(r.get("rejected_response"), str):
                        raise ValueError("DPO 缺少 rejected_response")
                    if task == "grpo" and not isinstance(r.get("solution"), str):
                        raise ValueError("数学 accuracy 奖励需要字符串 solution")
                    formats.add("messages")
                for key in ("images", "videos", "audios"):
                    if key in r:
                        if not isinstance(r[key], list):
                            raise ValueError(key + " 必须是文件路径列表")
                        for media in r[key]:
                            permitted(p.parent / media, roots)
                        formats.add(key)
            except (ValueError, KeyError, TypeError, OSError) as e:
                if len(errors) < 30:
                    errors.append({"line": i, "error": str(e)})
    if rows == 0:
        errors.append({"line": 0, "error": "数据为空"})
    return {"path": str(p), "task": task, "rows": rows, "formats": sorted(formats),
            "sha256": digest.hexdigest(), "errors": errors, "valid": not errors,
            "example": EXAMPLES[task], "note": "全文件结构检查，不代表质量、去重、版权及 tokenizer 长度验证通过"}
