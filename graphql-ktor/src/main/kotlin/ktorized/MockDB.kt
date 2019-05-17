package ktorized

import java.util.concurrent.atomic.AtomicInteger

// A toy domain class and database
data class ServerInfo(val name: String, val version: String, val peerIds: List<Int>)

object MockDB {
    private val data =  mutableMapOf(
            10 to ServerInfo("foo", "0.1.5", listOf(20, 30)),
            20 to ServerInfo("bar", "0.1.7", listOf(10)),
            30 to ServerInfo("baz", "0.1.2", listOf(10))
    )

    private var nextId = AtomicInteger(40)

    fun connect() = this
    fun fetch(id: Int) = data[id]

    fun insert(info: ServerInfo) {
        data[nextId.getAndAdd(10)] = info
    }
}
