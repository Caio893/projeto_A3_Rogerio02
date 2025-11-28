from fbx import *
import fbx
import sys


def _set_bool_prop(ios, prop_name, value):
    """
    Safely sets a boolean FBX IO property if the constant exists in the Python SDK.
    Some builds of the pip package expose fewer IMP/EXP_* constants; we skip silently
    when they are missing instead of raising NameError.
    """
    prop = getattr(fbx, prop_name, None)
    if prop is None:
        return False
    ios.SetBoolProp(prop, value)
    return True


def _set_string_prop(ios, prop_name, value):
    prop = getattr(fbx, prop_name, None)
    if prop is None:
        return False
    ios.SetStringProp(prop, value)
    return True

def InitializeSdkObjects():
    # The first thing to do is to create the FBX SDK manager which is the 
    # object allocator for almost all the classes in the SDK.
    lSdkManager = FbxManager.Create()
    if not lSdkManager:
        sys.exit(0)
        
    # Create an IOSettings object
    ios = FbxIOSettings.Create(lSdkManager, IOSROOT)
    lSdkManager.SetIOSettings(ios)
    
    # Create the entity that will hold the scene.
    lScene = FbxScene.Create(lSdkManager, "")
    
    return (lSdkManager, lScene)

def SaveScene(pSdkManager, pScene, pFilename, pFileFormat = -1, pEmbedMedia = False):
    lExporter = FbxExporter.Create(pSdkManager, "")
    if pFileFormat < 0 or pFileFormat >= pSdkManager.GetIOPluginRegistry().GetWriterFormatCount():
        pFileFormat = pSdkManager.GetIOPluginRegistry().GetNativeWriterFormat()
        if not pEmbedMedia:
            lFormatCount = pSdkManager.GetIOPluginRegistry().GetWriterFormatCount()
            for lFormatIndex in range(lFormatCount):
                if pSdkManager.GetIOPluginRegistry().WriterIsFBX(lFormatIndex):
                    lDesc = pSdkManager.GetIOPluginRegistry().GetWriterFormatDescription(lFormatIndex)
                    if "ascii" in lDesc:
                        pFileFormat = lFormatIndex
                        break
    
    if not pSdkManager.GetIOSettings():
        ios = FbxIOSettings.Create(pSdkManager, IOSROOT)
        pSdkManager.SetIOSettings(ios)
    
    ios = pSdkManager.GetIOSettings()
    _set_bool_prop(ios, "EXP_FBX_MATERIAL", True)
    _set_bool_prop(ios, "EXP_FBX_TEXTURE", True)
    _set_bool_prop(ios, "EXP_FBX_EMBEDDED", pEmbedMedia)
    _set_bool_prop(ios, "EXP_FBX_SHAPE", True)
    _set_bool_prop(ios, "EXP_FBX_GOBO", True)
    _set_bool_prop(ios, "EXP_FBX_ANIMATION", True)
    _set_bool_prop(ios, "EXP_FBX_GLOBAL_SETTINGS", True)

    result = lExporter.Initialize(pFilename, pFileFormat, pSdkManager.GetIOSettings())
    if result == True:
        result = lExporter.Export(pScene)

    lExporter.Destroy()
    return result
    
def LoadScene(pSdkManager, pScene, pFileName):
    lImporter = FbxImporter.Create(pSdkManager, "")    
    # Evita extra��o de texturas embutidas para pastas .fbm
    ios = pSdkManager.GetIOSettings()
    _set_bool_prop(ios, "IMP_FBX_EXTRACT_EMBEDDED_DATA", False)
    _set_string_prop(ios, "IMP_EXTRACT_FOLDER", "")
    result = lImporter.Initialize(pFileName, -1, ios)
    if not result:
        return False
    
    if lImporter.IsFBX():
        _set_bool_prop(ios, "EXP_FBX_MATERIAL", True)
        _set_bool_prop(ios, "EXP_FBX_TEXTURE", True)
        _set_bool_prop(ios, "EXP_FBX_EMBEDDED", True)
        _set_bool_prop(ios, "EXP_FBX_SHAPE", True)
        _set_bool_prop(ios, "EXP_FBX_GOBO", True)
        _set_bool_prop(ios, "EXP_FBX_ANIMATION", True)
        _set_bool_prop(ios, "EXP_FBX_GLOBAL_SETTINGS", True)
    
    result = lImporter.Import(pScene)
    lImporter.Destroy()
    return result
