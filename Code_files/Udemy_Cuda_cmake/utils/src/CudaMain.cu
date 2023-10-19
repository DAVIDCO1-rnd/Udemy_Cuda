// Nvcc predefines the macro __CUDACC__.
// This macro can be used in sources to test whether they are currently being compiled by nvcc.
#ifndef __CUDACC__
#error Must be compiled with CUDA compiler!
#endif


#include <cuda_runtime_api.h>
#include "CudaMain.cuh"

__wchar_t* CudaErrorToErrorMessage(cudaError err)
{
	switch (err)
	{
	case cudaSuccess:
		return NULL;
	case cudaErrorMissingConfiguration:
		return L"Missing configuration";
	case cudaErrorMemoryAllocation:
		return L"Memory allocation error";
	case cudaErrorInitializationError:
		return L"Initialization error";
	case cudaErrorLaunchFailure:
		return L"Launch failure";
	case cudaErrorPriorLaunchFailure:
		return L"Prior launch failure";
	case cudaErrorLaunchTimeout:
		return L"Launch timeout";
	case cudaErrorLaunchOutOfResources:
		return L"Launch out of resources";
	case cudaErrorInvalidDeviceFunction:
		return L"Invaild device function";
	case cudaErrorInvalidConfiguration:
		return L"Invalid configration";
	case cudaErrorInvalidDevice:
		return L"Invalid device";
	case cudaErrorInvalidValue:
		return L"Invalid value";
	case cudaErrorInvalidPitchValue:
		return L"Invalid pitch value";
	case cudaErrorInvalidSymbol:
		return L"Invalid symbol";
	case cudaErrorMapBufferObjectFailed:
		return L"Map of buffer object failed";
	case cudaErrorUnmapBufferObjectFailed:
		return L"Unmap of buffer object failed";
	case cudaErrorInvalidHostPointer:
		return L"Invalid (host) pointer";
	case cudaErrorInvalidDevicePointer:
		return L"Invalid GPU (device) pointer";
	case cudaErrorInvalidTexture:
		return L"Invalid texture";
	case cudaErrorInvalidTextureBinding:
		return L"Invalid texture binding";
	case cudaErrorInvalidChannelDescriptor:
		return L"Invalid channel descriptor";
	case cudaErrorInvalidMemcpyDirection:
		return L"Invalid memory copy direction";
	case cudaErrorAddressOfConstant:
		return L"Address of constant (!?)";
	case cudaErrorTextureFetchFailed:
		return L"Texture fetch failed";
	case cudaErrorTextureNotBound:
		return L"Texture not bound";
	case cudaErrorSynchronizationError:
		return L"Synchronization error";
	case cudaErrorInvalidFilterSetting:
		return L"Invalid filter setting";
	case cudaErrorInvalidNormSetting:
		return L"Invalid normal setting";
	case cudaErrorMixedDeviceExecution:
		return L"Mixed device execution";
	case cudaErrorCudartUnloading:
		return L"Cudart unloading (!?)";
	case cudaErrorUnknown:
		return L"Unknown error";
	case cudaErrorNotYetImplemented:
		return L"Not implemented yet";
	case cudaErrorMemoryValueTooLarge:
		return L"Memory value is too large";
	case cudaErrorInvalidResourceHandle:
		return L"Invalid resource handle";
	case cudaErrorNotReady:
		return L"Not ready... come back later";
	case cudaErrorSetOnActiveProcess:
		return L"cudaErrorSetOnActiveProcess";		
	case cudaErrorNoDevice:
		return L"cudaErrorNoDevice";		
	case cudaErrorDevicesUnavailable:
		return L"cudaErrorDevicesUnavailable";
	case cudaErrorStartupFailure:
		return L"Start up failure";
	case cudaErrorApiFailureBase:
		return L"API failure";
	default:
		return L"Other error";
	}
}

