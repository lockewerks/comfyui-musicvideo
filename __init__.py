"""comfyui-musicvideo: cut a music video to the beat, inside ComfyUI.

Analyse a song, plan an edit that cuts on bar lines, expand one master prompt
into a shot for every cut, render them and assemble the result against the
original audio.
"""

from .nodes.nodes_analysis import MVAnalyzeAudio, MVAudioCurve, MVBeatTimes
from .nodes.nodes_plan import MVShotPlan, MVPromptBook, MVShotInfo, MVPlanToJSON
from .nodes.nodes_render import (
    MVAssembleVideo,
    MVCollectImage,
    MVRenderShots,
    MVRenderStills,
    MVShotsBegin,
    MVWriteShot,
)

NODE_CLASS_MAPPINGS = {
    "MVAnalyzeAudio": MVAnalyzeAudio,
    "MVAudioCurve": MVAudioCurve,
    "MVBeatTimes": MVBeatTimes,
    "MVShotPlan": MVShotPlan,
    "MVPromptBook": MVPromptBook,
    "MVShotInfo": MVShotInfo,
    "MVPlanToJSON": MVPlanToJSON,
    "MVRenderStills": MVRenderStills,
    "MVRenderShots": MVRenderShots,
    "MVAssembleVideo": MVAssembleVideo,
    # Internal, created by the expansion nodes rather than placed by hand.
    "MVShotsBegin": MVShotsBegin,
    "MVWriteShot": MVWriteShot,
    "MVCollectImage": MVCollectImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MVAnalyzeAudio": "Analyse audio",
    "MVAudioCurve": "Audio curve",
    "MVBeatTimes": "Beat times",
    "MVShotPlan": "Shot plan",
    "MVPromptBook": "Prompt book",
    "MVShotInfo": "Shot info",
    "MVPlanToJSON": "Shot plan to JSON",
    "MVRenderStills": "Render start frames",
    "MVRenderShots": "Render shots",
    "MVAssembleVideo": "Assemble music video",
    "MVShotsBegin": "Shot collection (begin)",
    "MVWriteShot": "Write shot",
    "MVCollectImage": "Collect image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
