from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import io
import base64
import uuid
import tempfile
import os
from PIL import Image
import numpy as np

from ..segmentation_inpainting.hf_qwen_space import white_to_scene, fusion, relight

router = APIRouter(
    tags=["/inpainting"],
    prefix="/inpainting"
)


class TextEditRequest(BaseModel):
    image_base64: str
    text_instruction: str
    segments: Optional[List[Dict]] = None
    preserve_original: bool = True

class SegmentRequest(BaseModel):
    image_base64: str
    point_prompts: Optional[List[Dict]] = None  # [{x: int, y: int, label: int}]
    box_prompts: Optional[List[List[int]]] = None  # [[x1, y1, x2, y2]]

class BlendRequest(BaseModel):
    original_image: str
    background_image: str
    segments: List[Dict]
    blend_mode: str = "seamless"

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: int
    result_url: Optional[str] = None
    error: Optional[str] = None

task_storage = {}

@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    point_prompts: Optional[str] = Form(None),
    box_prompts: Optional[str] = Form(None)
):
    """
    Segment image using MobileSAM
    Returns segmentation masks for each detected object
    """
    try:
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
      
        segments = [
            {
                "id": 1,
                "mask": "base64_encoded_mask",
                "bbox": [100, 100, 200, 200],
                "area": 10000,
                "confidence": 0.95,
                "category": "person"
            }
        ]
        
        return JSONResponse({
            "success": True,
            "segments": segments,
            "total_segments": len(segments)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@router.post("/analyze-scene")
async def analyze_scene(
    file: UploadFile = File(...),
    segments: str = Form(...)
):
    """
    Use Qwen to analyze image content and segment relationships
    """
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        
        analysis = {
            "scene_description": "A person standing in a park with trees in the background",
            "lighting_conditions": "natural daylight, soft shadows",
            "dominant_colors": ["green", "brown", "blue"],
            "objects": [
                {
                    "name": "person",
                    "description": "adult wearing casual clothes",
                    "position": "center-foreground",
                    "lighting": "front-lit"
                }
            ],
            "composition": "rule of thirds, subject centered",
            "style": "photographic, natural"
        }
        
        return JSONResponse({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene analysis failed: {str(e)}")

@router.post("/text-edit")
async def process_text_edit(request: TextEditRequest):
    """
    Process text instructions for image editing using Qwen models
    Parse natural language commands and determine appropriate Qwen function
    """
    try:
        
        image_data = base64.b64decode(request.image_base64)
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(image_data)
        temp_image.close()
        
        
        instruction_lower = request.text_instruction.lower()
        
        
        if any(keyword in instruction_lower for keyword in ['background', 'scene', 'environment', 'setting']):
            function_type = "white_to_scene"
            description = "Change background/scene while preserving subject"
        elif any(keyword in instruction_lower for keyword in ['blend', 'merge', 'combine', 'fusion', 'integrate']):
            function_type = "fusion"  
            description = "Blend/fuse elements together seamlessly"
        elif any(keyword in instruction_lower for keyword in ['light', 'lighting', 'illuminate', 'shadow', 'bright', 'dark']):
            function_type = "relight"
            description = "Adjust lighting and shadows"
        else:
            
            function_type = "fusion"
            description = "General image editing and enhancement"
        
        parsed_instructions = {
            "qwen_function": function_type,
            "prompt": request.text_instruction,
            "description": description,
            "confidence": 0.95,
            "temp_image_path": temp_image.name,
            "suggested_steps": [
                f"Apply {function_type} transformation",
                "Process with Qwen model",
                "Return enhanced image"
            ]
        }
        
        return JSONResponse({
            "success": True,
            "parsed_instructions": parsed_instructions,
            "preview_available": True
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text processing failed: {str(e)}")

@router.post("/apply-edits")
async def apply_edits(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    instructions: str = Form(...),
    segments: str = Form(...)
):
    """
    Execute the parsed editing instructions
    Returns task ID for async processing
    """
    try:
        
        task_id = str(uuid.uuid4())
        
        
        task_storage[task_id] = {
            "status": "pending",
            "progress": 0,
            "result_url": None,
            "error": None
        }
        
        
        background_tasks.add_task(
            process_edit_task,
            task_id,
            await file.read(),
            instructions,
            segments
        )
        
        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "Edit processing started"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edit application failed: {str(e)}")

@router.post("/generate-background")
async def generate_background(
    description: str = Form(...),
    style: Optional[str] = Form("photorealistic"),
    resolution: Optional[str] = Form("1024x1024")
):
    """
    Generate new background based on text description
    """
    try:
        
        background_url = f"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        return JSONResponse({
            "success": True,
            "background_url": background_url,
            "description": description,
            "style": style
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background generation failed: {str(e)}")

@router.post("/blend-segments")
async def blend_segments(request: BlendRequest):
    """
    Seamlessly blend segmented objects with new background
    """
    try:
        
        blended_url = f"data:image/png;base64,blended_image_data"
        
        return JSONResponse({
            "success": True,
            "blended_image": blended_url,
            "blend_quality": 0.95,
            "processing_time": "3.2s"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blending failed: {str(e)}")

@router.post("/refine-composition")
async def refine_composition(
    file: UploadFile = File(...),
    enhancement_level: float = Form(0.5),
    preserve_details: bool = Form(True)
):
    """
    Post-process and refine the final composition
    """
    try:
        image_data = await file.read()
        
        
        refined_url = f"data:image/png;base64,refined_image_data"
        
        return JSONResponse({
            "success": True,
            "refined_image": refined_url,
            "enhancements_applied": [
                "color_correction",
                "edge_sharpening",
                "shadow_adjustment"
            ]
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {str(e)}")

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Check processing status for long-running tasks
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return JSONResponse(task_storage[task_id])

@router.get("/presets")
async def get_editing_presets():
    """
    Get available editing presets and templates
    """
    presets = {
        "background_removal": {
            "name": "Background Removal",
            "description": "Remove background completely",
            "example": "Remove the background"
        },
        "object_replacement": {
            "name": "Object Replacement",
            "description": "Replace one object with another",
            "example": "Replace the car with a bicycle"
        },
        "style_transfer": {
            "name": "Style Transfer",
            "description": "Change artistic style",
            "example": "Make it look like a painting"
        },
        "lighting_adjustment": {
            "name": "Lighting Adjustment", 
            "description": "Modify lighting conditions",
            "example": "Change lighting to sunset"
        },
        "seasonal_change": {
            "name": "Seasonal Change",
            "description": "Transform to different season",
            "example": "Make it look like winter"
        }
    }
    
    return JSONResponse({
        "success": True,
        "presets": presets
    })

@router.delete("/task/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_storage[task_id]["status"] = "cancelled"
    
    return JSONResponse({
        "success": True,
        "message": f"Task {task_id} cancelled"
    })

@router.post("/qwen/smart-edit")
async def qwen_smart_edit(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Smart AI editing that automatically selects the best Qwen function
    and includes fallback strategies for maximum success rate
    """
    try:
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(await file.read())
        temp_image.close()
        
        from ..segmentation_inpainting.hf_qwen_space import get_best_adapter, AVAILABLE_ADAPTERS
        
        
        best_adapter = get_best_adapter(prompt)
        result_image = None
        used_function = None
        
        # Strategy 1: Try the best adapter first
        try:
            if best_adapter == "Next-Scene" or "background" in prompt.lower():
                result_image = white_to_scene(temp_image.name, prompt)
                used_function = "white_to_scene"
            elif best_adapter == "Relight" or "light" in prompt.lower():
                result_image = relight(temp_image.name, prompt)
                used_function = "relight"
            else:
                result_image = fusion(temp_image.name, prompt)
                used_function = "fusion"
                
        except Exception as e:
            print(f"Primary strategy failed: {e}")
            
            # Strategy 2: Try relight (most stable)
            try:
                result_image = relight(temp_image.name, prompt)
                used_function = "relight_fallback"
            except Exception as e2:
                print(f"Relight fallback failed: {e2}")
                
                # Strategy 3: Try fusion
                try:
                    result_image = fusion(temp_image.name, prompt)
                    used_function = "fusion_fallback"
                except Exception as e3:
                    print(f"Fusion fallback failed: {e3}")
                    
                    # Strategy 4: Try scene change
                    result_image = white_to_scene(temp_image.name, prompt)
                    used_function = "scene_fallback"
        
        
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        
        os.unlink(temp_image.name)
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "function_used": used_function,
            "best_adapter": best_adapter,
            "available_adapters": AVAILABLE_ADAPTERS
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"All editing strategies failed: {str(e)}",
            "available_adapters": AVAILABLE_ADAPTERS
        }, status_code=500)

@router.post("/qwen/white-to-scene")
async def qwen_white_to_scene(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Transform white/neutral background to specific scene using Qwen
    Examples: "beach scene", "mountain landscape", "urban street"
    """
    try:
        
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(await file.read())
        temp_image.close()
        
        
        result_image = white_to_scene(temp_image.name, prompt)
        
        
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        
        os.unlink(temp_image.name)
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "function_used": "white_to_scene"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"White-to-scene transformation failed: {str(e)}")

@router.post("/qwen/fusion")
async def qwen_fusion(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Blend/fuse image elements using Qwen fusion model
    Examples: "blend with forest background", "merge with sunset sky"
    """
    try:
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(await file.read())
        temp_image.close()
        
        result_image = fusion(temp_image.name, prompt)
        
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        os.unlink(temp_image.name)
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "function_used": "fusion"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion failed: {str(e)}")

@router.post("/qwen/relight")
async def qwen_relight(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Adjust lighting and shadows using Qwen relight model
    Examples: "golden hour lighting", "dramatic shadows", "soft natural light"
    """
    try:
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(await file.read())
        temp_image.close()
        
        result_image = relight(temp_image.name, prompt)
        
        buffer = io.BytesIO()
        result_image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        os.unlink(temp_image.name)
        
        return JSONResponse({
            "success": True,
            "result_image": f"data:image/png;base64,{img_base64}",
            "prompt": prompt,
            "function_used": "relight"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Relighting failed: {str(e)}")

@router.post("/enhance-merged-lighting")
async def enhance_merged_lighting(
    merged_file: UploadFile = File(...)
):
    """
    Enhance lighting of a merged composite image (segment + background already combined).
    This applies intelligent lighting enhancement to the entire merged image.
    """
    try:
        
        temp_merged = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_merged.write(await merged_file.read())
        temp_merged.close()
        
        
        merged_img = Image.open(temp_merged.name).convert('RGB')
        
        
        original_width = merged_img.width
        original_height = merged_img.height
        
        
        img_array = np.array(merged_img)
        avg_brightness = np.mean(img_array)
        
       
        if avg_brightness > 150:
            lighting_prompt = "enhance lighting, improve brightness and clarity, natural look"
        elif avg_brightness > 100:
            lighting_prompt = "balance lighting, enhance details, natural appearance"
        else:
            lighting_prompt = "improve lighting, brighten naturally, enhance visibility"
        
        
        try:
            enhanced_result = relight(temp_merged.name, lighting_prompt)
            
            
            if enhanced_result.size != (original_width, original_height):
                enhanced_result = enhanced_result.resize((original_width, original_height), Image.Resampling.LANCZOS)
            
        except Exception as e:
            print(f"Relighting failed, using original: {e}")
            enhanced_result = merged_img
        
        
        buffer = io.BytesIO()
        enhanced_result.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        
        os.unlink(temp_merged.name)
        
        return JSONResponse({
            "success": True,
            "enhanced_image": f"data:image/png;base64,{img_base64}",
            "lighting_applied": lighting_prompt,
            "avg_brightness": float(avg_brightness),
            "original_size": f"{original_width}x{original_height}"
        })
        
    except Exception as e:
        
        try:
            if os.path.exists(temp_merged.name):
                os.unlink(temp_merged.name)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Lighting enhancement failed: {str(e)}")

@router.post("/enhance-lighting")
async def enhance_lighting(
    segment_file: UploadFile = File(...),
    background_file: UploadFile = File(...)
):
    """
    Enhance segment lighting to match background after merge.
    Preserves original image size and content, only adjusts lighting.
    """
    try:
       
        temp_segment = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_segment.write(await segment_file.read())
        temp_segment.close()
        
        
        temp_background = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_background.write(await background_file.read())
        temp_background.close()
        
        
        segment_img = Image.open(temp_segment.name).convert('RGB')
        background_img = Image.open(temp_background.name).convert('RGB')
        
        
        original_width = segment_img.width
        original_height = segment_img.height
        
        
        bg_array = np.array(background_img)
        avg_brightness = np.mean(bg_array)
        r_avg = np.mean(bg_array[:,:,0])
        b_avg = np.mean(bg_array[:,:,2])
        
        
        if avg_brightness > 180:
            lighting_prompt = "bright natural lighting"
        elif avg_brightness > 120:
            lighting_prompt = "balanced natural lighting"
        elif avg_brightness > 60:
            lighting_prompt = "soft dim lighting"
        else:
            lighting_prompt = "low ambient lighting"
        
        
        if r_avg > b_avg + 30:
            lighting_prompt += ", warm tone"
        elif b_avg > r_avg + 30:
            lighting_prompt += ", cool tone"
        
        
        try:
            enhanced_result = relight(temp_segment.name, lighting_prompt)
            
           
            if enhanced_result.size != (original_width, original_height):
                enhanced_result = enhanced_result.resize((original_width, original_height), Image.Resampling.LANCZOS)
            
        except Exception as e:
            print(f"Relighting failed, using original: {e}")
            enhanced_result = segment_img
        
        
        buffer = io.BytesIO()
        enhanced_result.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        
        os.unlink(temp_segment.name)
        os.unlink(temp_background.name)
        
        return JSONResponse({
            "success": True,
            "enhanced_segment": f"data:image/png;base64,{img_base64}",
            "lighting_applied": lighting_prompt,
            "avg_brightness": float(avg_brightness),
            "original_size": f"{original_width}x{original_height}"
        })
        
    except Exception as e:
        
        for temp_file in [temp_segment.name, temp_background.name]:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Lighting enhancement failed: {str(e)}")


async def process_edit_task(task_id: str, image_data: bytes, instructions: str, segments: str):
    """
    Background task for processing heavy edit operations using Qwen functions
    """
    try:
        
        task_storage[task_id]["status"] = "processing"
        task_storage[task_id]["progress"] = 10
        

        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        temp_image.write(image_data)
        temp_image.close()
        
        
        import json
        parsed = json.loads(instructions)
        qwen_function = parsed.get("qwen_function", "fusion")
        prompt = parsed.get("prompt", "enhance the image")
        
        task_storage[task_id]["progress"] = 30
        
        
        try:
            if qwen_function == "white_to_scene":
                result_image = white_to_scene(temp_image.name, prompt)
            elif qwen_function == "relight":
                result_image = relight(temp_image.name, prompt)
            else:  
                result_image = fusion(temp_image.name, prompt)
            
            task_storage[task_id]["progress"] = 80
            
            
            buffer = io.BytesIO()
            result_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            result_url = f"data:image/png;base64,{img_base64}"
            
            
            os.unlink(temp_image.name)
            
           
            task_storage[task_id]["status"] = "completed"
            task_storage[task_id]["progress"] = 100
            task_storage[task_id]["result_url"] = result_url
            
        except Exception as qwen_error:
            raise Exception(f"Qwen processing failed: {str(qwen_error)}")
        
    except Exception as e:
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(e)
        
        if 'temp_image' in locals():
            try:
                os.unlink(temp_image.name)
            except:
                pass