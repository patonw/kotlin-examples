/*
 * This is mainly translated & adapted from http://wiki.lwjgl.org/wiki/OpenCL_in_LWJGL.html
 * I've added some helper classes and extension functions in CLUtil.kt to make the example easier to follow
 */
package lwocl

import org.lwjgl.BufferUtils
import org.lwjgl.opencl.CL10.*
import java.nio.FloatBuffer

val sumKernelSource = """
    kernel void sum(global const float* a, global const float* b, global float* result, int const size) {
      const int itemId = get_global_id(0);
      if(itemId < size) {
        result[itemId] = a[itemId] + b[itemId] * 2;
      }
    }
""".trimIndent()

class SumApp(val ctx: CLContext) {
    private val nElems = 1024

    private// Create float array from 0 to nElems-1.
    val aBuffer: FloatBuffer by lazy {
            val aBuff = BufferUtils.createFloatBuffer(nElems)
            val tempData = FloatArray(nElems)
            for (i in 0 until nElems) {
                tempData[i] = i.toFloat()
                println("a[$i]=$i")
            }
            aBuff.put(tempData)
            aBuff.rewind()
            aBuff
        }

    private// Create float array from nElems-1 to 0. This means that the result should be nElems-1 for each element.
    val bBuffer: FloatBuffer by lazy {
            val bBuff = BufferUtils.createFloatBuffer(nElems)
            val tempData = FloatArray(nElems)
            var j = 0
            var i = nElems - 1
            while (j < nElems) {
                tempData[j] = i.toFloat()
                println("b[$j]=$i")
                j++
                i--
            }
            bBuff.put(tempData)
            bBuff.rewind()
            bBuff
        }

    fun run() {
        arrayOf(ctx.createBuffer(aBuffer), ctx.createBuffer(bBuffer), ctx.createResultBuffer((nElems * 4L))).use { (a, b, result) ->
            ctx.makeKernel(sumKernelSource, "sum").use { kernel ->
                kernel.apply {
                    setArg(0, a)
                    setArg(1, b)
                    setArg(2, result)
                    setArg(3, nElems)
                }

                val dimensions = 1
                val globalWorkSize = makeGlobalWorkSize(nElems.toLong())

                // Coroutine wrapper might be useful for this part
                // Run the specified number of work units using our OpenCL program kernel
                checkCLError(clEnqueueNDRangeKernel(ctx.queue, kernel.handle, dimensions, null, globalWorkSize, null, null, null))

                clFinish(ctx.queue)

                printResults(result)
            }
        }
    }

    private fun printResults(resultMemory: CLPointer) {
        // This reads the result memory buffer
        val resultBuff = BufferUtils.createFloatBuffer(nElems)
        // We read the buffer in blocking mode so that when the method returns we know that the result buffer is full
        clEnqueueReadBuffer(ctx.queue, resultMemory.handle, true, 0, resultBuff, null, null)
        clFinish(ctx.queue)
        // Print the values in the result buffer
        for (i in 0 until resultBuff.capacity()) {
            println("result at " + i + " = " + resultBuff.get(i))
        }
        // This should print out 100 lines of result floats, each being 99.
    }
}

fun main() {
    CLContext().use { ctx ->
        SumApp(ctx).run()
    }
}