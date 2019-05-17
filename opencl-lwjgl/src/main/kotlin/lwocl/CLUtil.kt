package lwocl

import org.lwjgl.BufferUtils
import org.lwjgl.PointerBuffer
import org.lwjgl.opencl.CL
import org.lwjgl.opencl.CL10
import org.lwjgl.opencl.CLCapabilities
import org.lwjgl.system.MemoryStack
import org.lwjgl.system.MemoryUtil
import java.io.Closeable
import java.nio.FloatBuffer
import java.nio.IntBuffer


data class OpenCLPlatform(val id: Long, val capabilities: CLCapabilities)

fun checkCLError(errCode: Int) {
    if (errCode != CL10.CL_SUCCESS) {
        throw RuntimeException(String.format("OpenCL error [%d]", errCode));
    }
}

typealias ErrorBuffer = IntBuffer

fun ErrorBuffer.checkAndClear() {
    checkCLError(get())
    clear()
}

data class CLPointer(val handle: Long) : Closeable {
    override fun close() {
        CL10.clReleaseMemObject(handle)
    }
}

open class CLContext : Closeable {
    val errBuf: ErrorBuffer = BufferUtils.createIntBuffer(1)
    var handle: Long = 0
    var device: Long = 0
    var queue: Long = 0

    init {
        initializeCL()
    }

    override fun close() {
        CL10.clReleaseCommandQueue(queue)

        // Not strictly necessary
        CL.destroy()
    }

    fun initializeCL() {
        val (clPlatform: Long, clPlatformCapabilities: CLCapabilities) = getFirstPlatform()

        device = getDevice(clPlatform, clPlatformCapabilities, CL10.CL_DEVICE_TYPE_GPU)

        // Create the context
        val ctxProps = BufferUtils.createPointerBuffer(7)
        ctxProps.put(CL10.CL_CONTEXT_PLATFORM.toLong()).put(clPlatform).put(MemoryUtil.NULL).flip()

        handle = CL10.clCreateContext(
                ctxProps,
                device,
                { errinfo, private_info, cb, user_data ->
                    System.out.printf("cl_context_callback\n\tInfo: %s", MemoryUtil.memUTF8(errinfo))
                },
                MemoryUtil.NULL,
                errBuf
        )

        // create command queue
        queue = CL10.clCreateCommandQueue(handle, device, MemoryUtil.NULL, errBuf)
        errBuf.checkAndClear()
    }

    // TODO Operate on sequence of all platforms. Use map/filter/first
    fun getFirstPlatform(): OpenCLPlatform {
        // Get the first available platform
        MemoryStack.stackPush().use { stack ->
            val pi = stack.mallocInt(1)
            checkCLError(CL10.clGetPlatformIDs(null, pi))
            if (pi.get(0) == 0) {
                throw IllegalStateException("No OpenCL platforms found.")
            }

            val platformIDs = stack.mallocPointer(pi.get(0))
            checkCLError(CL10.clGetPlatformIDs(platformIDs, null as IntBuffer?))

            if (platformIDs.capacity() > 0) {
                val platform = platformIDs.get(0)
                val clPlatformCapabilities = CL.createPlatformCapabilities(platform)
                return OpenCLPlatform(platform, clPlatformCapabilities)
            }
        }

        throw RuntimeException("No Platforms available")
    }

    fun getDevice(platform: Long, platformCaps: CLCapabilities?, deviceType: Int): Long {
        MemoryStack.stackPush().use { stack ->
            val pi = stack.mallocInt(1)
            checkCLError(CL10.clGetDeviceIDs(platform, deviceType.toLong(), null, pi))

            val devices = stack.mallocPointer(pi.get(0))
            checkCLError(CL10.clGetDeviceIDs(platform, deviceType.toLong(), devices, null as IntBuffer?))

            for (i in 0 until devices.capacity()) {
                val device = devices.get(i)

                val caps = CL.createDeviceCapabilities(device, platformCaps!!)
                if (!(caps.cl_khr_gl_sharing || caps.cl_APPLE_gl_sharing)) {
                    continue
                }

                return device
            }
        }

        return MemoryUtil.NULL
    }
}

class CLKernel private constructor(val sumProgram: Long, val handle: Long) : Closeable {
    companion object {
        fun make(ctx: CLContext, source: String, entryPoint: String) : CLKernel {
            val errBuf: ErrorBuffer = BufferUtils.createIntBuffer(1)
            val sumProgram = CL10.clCreateProgramWithSource(ctx.handle, source, errBuf)
            errBuf.checkAndClear()
            checkCLError(CL10.clBuildProgram(sumProgram, ctx.device, "", null, MemoryUtil.NULL))
            val handle = CL10.clCreateKernel(sumProgram, entryPoint, errBuf)
            errBuf.checkAndClear()
            return CLKernel(sumProgram, handle)
        }
    }

    override fun close() {
        // Destroy our kernel and program
        CL10.clReleaseKernel(handle)
        CL10.clReleaseProgram(sumProgram)
    }

    fun setArg(i: Int, value: Int) = CL10.clSetKernelArg1i(handle, i, value)
    fun setArg(i: Int, ptr: CLPointer) = CL10.clSetKernelArg1p(handle, i, ptr.handle)
}

fun makeGlobalWorkSize(vararg dims: Long) = BufferUtils.createPointerBuffer(dims.size).apply {
    dims.forEachIndexed { i,d ->
        put(i, d)
    }
}
fun CLContext.makeKernel(source: String, entryPoint: String) = CLKernel.make(this, source, entryPoint)

fun CLContext.createBuffer(data: FloatBuffer, flags: Int = (CL10.CL_MEM_WRITE_ONLY or CL10.CL_MEM_COPY_HOST_PTR)): CLPointer {
    val result = CLPointer(CL10.clCreateBuffer(handle, flags.toLong(), data, errBuf))
    errBuf.checkAndClear()
    return result
}

fun CLContext.createResultBuffer(size: Long, flags: Int = CL10.CL_MEM_READ_ONLY): CLPointer {
    val result = CLPointer(CL10.clCreateBuffer(handle, flags.toLong(), size, errBuf))
    errBuf.checkAndClear()
    return result
}

// Ensure that all closeable resources are released when block completes
inline fun <T: Closeable> Array<T>.use(block: (Array<T>) -> Unit) {
    try {
        block(this)
    }

    finally {
        for (a in this) {
            try {
                a.close()
            }
            catch(e:Exception) {
            }
        }
    }
}