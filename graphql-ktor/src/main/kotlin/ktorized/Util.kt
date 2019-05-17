package ktorized

import graphql.schema.DataFetchingEnvironment
import graphql.schema.GraphQLSchema
import graphql.schema.idl.RuntimeWiring
import graphql.schema.idl.SchemaGenerator
import graphql.schema.idl.TypeDefinitionRegistry
import graphql.schema.idl.TypeRuntimeWiring

// Convenience extension to hide some indirection in the graphql-java API
fun TypeDefinitionRegistry.wiring(block: RuntimeWiring.Builder.() -> Unit): GraphQLSchema {
    val wiring = RuntimeWiring.newRuntimeWiring().apply(block).build()
    return SchemaGenerator().makeExecutableSchema(this, wiring)
}
