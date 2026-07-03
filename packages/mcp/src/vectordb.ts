import {
    VectorDatabase,
    MilvusVectorDatabase,
    ChromaRestfulVectorDatabase
} from "@zilliz/claude-context-core";
import { ContextMcpConfig } from "./config.js";

// Helper function to create a vector database instance based on the configured provider
export function createVectorDatabaseInstance(config: ContextMcpConfig): VectorDatabase {
    console.log(`[VECTORDB] Creating ${config.vectorDatabaseProvider} vector database instance...`);

    switch (config.vectorDatabaseProvider) {
        case 'Chroma': {
            console.log(`[VECTORDB] 🔧 Configuring ChromaDB at: ${config.chromaAddress}`);
            const chroma = new ChromaRestfulVectorDatabase({
                address: config.chromaAddress,
                ...(config.chromaToken && { token: config.chromaToken }),
                ...(config.chromaTenant && { tenant: config.chromaTenant }),
                ...(config.chromaDatabase && { database: config.chromaDatabase })
            });
            console.log(`[VECTORDB] ✅ ChromaDB instance created successfully`);
            return chroma;
        }

        case 'Milvus':
        default: {
            console.log(`[VECTORDB] 🔧 Configuring Milvus at: ${config.milvusAddress || '[Auto-resolve from token]'}`);
            const milvus = new MilvusVectorDatabase({
                address: config.milvusAddress,
                ...(config.milvusToken && { token: config.milvusToken })
            });
            console.log(`[VECTORDB] ✅ Milvus instance created successfully`);
            return milvus;
        }
    }
}
