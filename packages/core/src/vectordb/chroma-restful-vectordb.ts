/**
 * ChromaDB RESTful Vector Database Implementation
 *
 * This RESTful implementation of ChromaDB is designed to mirror the
 * MilvusRestfulVectorDatabase implementation so ChromaDB can be used as a
 * drop-in `VectorDatabase` backend. It relies only on HTTP `fetch`, making it
 * usable in dependency-constrained environments (VSCode / Chrome extensions).
 *
 * It talks to Chroma's v2 REST API:
 *   /api/v2/tenants/{tenant}/databases/{database}/collections...
 *
 * Notes on the mapping between the `VectorDocument` model and Chroma:
 *  - `content`      -> Chroma `documents`
 *  - `vector`       -> Chroma `embeddings`
 *  - everything else (relativePath, startLine, endLine, fileExtension and the
 *    caller-provided `metadata` object serialized as JSON) is stored in the
 *    per-record Chroma `metadatas` map, whose values must be scalars.
 *
 * Chroma keys every data operation on a collection UUID rather than the
 * collection name, so we resolve and cache the name -> id mapping.
 */

import {
    VectorDocument,
    SearchOptions,
    VectorSearchResult,
    VectorDatabase,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult,
} from './types';

export interface ChromaRestfulConfig {
    /** Base URL of the Chroma server, e.g. http://localhost:8000 */
    address?: string;
    /** Optional bearer token / api key for authenticated deployments */
    token?: string;
    /** Chroma tenant (defaults to "default_tenant") */
    tenant?: string;
    /** Chroma database (defaults to "default_database") */
    database?: string;
}

const DEFAULT_TENANT = 'default_tenant';
const DEFAULT_DATABASE = 'default_database';

/**
 * ChromaDB Vector Database implementation using its REST API.
 */
export class ChromaRestfulVectorDatabase implements VectorDatabase {
    protected config: ChromaRestfulConfig;
    private baseUrl: string | null = null;
    protected initializationPromise: Promise<void>;

    /** Cache of collectionName -> Chroma collection UUID */
    private collectionIdCache: Map<string, string> = new Map();

    constructor(config: ChromaRestfulConfig) {
        this.config = config;
        this.initializationPromise = this.initialize();
    }

    private async initialize(): Promise<void> {
        let address = this.config.address;
        if (!address) {
            throw new Error('ChromaDB address is required (e.g. http://localhost:8000)');
        }

        if (!address.startsWith('http://') && !address.startsWith('https://')) {
            address = `http://${address}`;
        }
        address = address.replace(/\/$/, '');

        const tenant = this.config.tenant || DEFAULT_TENANT;
        const database = this.config.database || DEFAULT_DATABASE;

        this.baseUrl = `${address}/api/v2/tenants/${encodeURIComponent(tenant)}/databases/${encodeURIComponent(database)}`;

        console.log(`🔌 Connecting to ChromaDB REST API at: ${address} (tenant=${tenant}, database=${database})`);
    }

    protected async ensureInitialized(): Promise<void> {
        await this.initializationPromise;
        if (!this.baseUrl) {
            throw new Error('Base URL not initialized');
        }
    }

    /**
     * Make an HTTP request to the Chroma REST API.
     * `endpoint` is relative to the tenant/database base URL.
     */
    private async makeRequest(endpoint: string, method: 'GET' | 'POST' | 'DELETE' = 'POST', data?: any): Promise<any> {
        const url = `${this.baseUrl}${endpoint}`;

        const headers: Record<string, string> = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        };

        if (this.config.token) {
            headers['Authorization'] = `Bearer ${this.config.token}`;
        }

        const requestOptions: RequestInit = { method, headers };
        if (data !== undefined && method !== 'GET') {
            requestOptions.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, requestOptions);

            if (!response.ok) {
                let detail = response.statusText;
                try {
                    const body: any = await response.json();
                    detail = body?.message || body?.error || JSON.stringify(body) || detail;
                } catch {
                    // ignore body parse errors
                }
                throw new Error(`HTTP ${response.status}: ${detail}`);
            }

            // Some endpoints (DELETE) may return an empty body.
            const text = await response.text();
            return text ? JSON.parse(text) : {};
        } catch (error) {
            console.error(`[ChromaRestfulDB] ChromaDB REST API request failed (${method} ${endpoint}):`, error);
            throw error;
        }
    }

    /**
     * Resolve (and cache) the Chroma UUID for a collection name.
     * Throws if the collection does not exist.
     */
    private async getCollectionId(collectionName: string): Promise<string> {
        const cached = this.collectionIdCache.get(collectionName);
        if (cached) return cached;

        // Chroma v2 accepts the collection name in the get-by-name path.
        const response = await this.makeRequest(`/collections/${encodeURIComponent(collectionName)}`, 'GET');
        const id = response?.id;
        if (!id) {
            throw new Error(`Collection '${collectionName}' not found`);
        }
        this.collectionIdCache.set(collectionName, id);
        return id;
    }

    async createCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        await this.ensureInitialized();

        try {
            // `dimension` is not required up-front by Chroma (it is inferred from the
            // first inserted embedding), but we keep the signature for interface parity.
            const body = {
                name: collectionName,
                metadata: {
                    description: description || `Claude Context collection: ${collectionName}`,
                    'hnsw:space': 'cosine'
                },
                configuration: {
                    hnsw: { space: 'cosine' }
                },
                get_or_create: false
            };

            const response = await this.makeRequest('/collections', 'POST', body);
            if (response?.id) {
                this.collectionIdCache.set(collectionName, response.id);
            }
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to create collection '${collectionName}':`, error);
            throw error;
        }
    }

    async createHybridCollection(collectionName: string, dimension: number, description?: string): Promise<void> {
        // ChromaDB has no native BM25 sparse-vector hybrid index. We create a
        // regular dense collection; hybridSearch() degrades to dense search.
        console.warn('[ChromaRestfulDB] ⚠️  ChromaDB does not support native hybrid (sparse) search; creating a dense-only collection.');
        await this.createCollection(collectionName, dimension, description);
    }

    async dropCollection(collectionName: string): Promise<void> {
        await this.ensureInitialized();

        try {
            await this.makeRequest(`/collections/${encodeURIComponent(collectionName)}`, 'DELETE');
            this.collectionIdCache.delete(collectionName);
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to drop collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hasCollection(collectionName: string): Promise<boolean> {
        await this.ensureInitialized();

        try {
            await this.getCollectionId(collectionName);
            return true;
        } catch (error) {
            return false;
        }
    }

    async listCollections(): Promise<string[]> {
        await this.ensureInitialized();

        try {
            const response = await this.makeRequest('/collections', 'GET');
            const collections: any[] = Array.isArray(response) ? response : (response?.collections || []);
            // Refresh the id cache while we are here.
            for (const c of collections) {
                if (c?.name && c?.id) this.collectionIdCache.set(c.name, c.id);
            }
            return collections.map((c: any) => c.name).filter(Boolean);
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to list collections:`, error);
            throw error;
        }
    }

    /**
     * Build the Chroma metadatas map for a document. All values must be scalars,
     * so the caller-provided metadata object is serialized as a JSON string.
     */
    private toChromaMetadata(doc: VectorDocument): Record<string, any> {
        return {
            relativePath: doc.relativePath,
            startLine: doc.startLine,
            endLine: doc.endLine,
            fileExtension: doc.fileExtension,
            metadata: JSON.stringify(doc.metadata ?? {})
        };
    }

    async insert(collectionName: string, documents: VectorDocument[]): Promise<void> {
        await this.ensureInitialized();

        if (documents.length === 0) return;

        try {
            const collectionId = await this.getCollectionId(collectionName);

            const body = {
                ids: documents.map(d => d.id),
                embeddings: documents.map(d => d.vector),
                documents: documents.map(d => d.content),
                metadatas: documents.map(d => this.toChromaMetadata(d))
            };

            await this.makeRequest(`/collections/${collectionId}/add`, 'POST', body);
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to insert documents into collection '${collectionName}':`, error);
            throw error;
        }
    }

    async insertHybrid(collectionName: string, documents: VectorDocument[]): Promise<void> {
        // No sparse vector to store; identical to dense insert.
        await this.insert(collectionName, documents);
    }

    /**
     * Reconstruct a VectorDocument from a Chroma metadata map + document text.
     */
    private fromChromaRecord(id: string, content: string, meta: Record<string, any> | null, vector: number[]): VectorDocument {
        const m = meta || {};
        let metadata: Record<string, any> = {};
        try {
            metadata = typeof m.metadata === 'string' ? JSON.parse(m.metadata || '{}') : (m.metadata || {});
        } catch (error) {
            console.warn(`[ChromaRestfulDB] Failed to parse metadata for item ${id}:`, error);
            metadata = {};
        }

        return {
            id: id?.toString() || '',
            vector,
            content: content || '',
            relativePath: m.relativePath || '',
            startLine: m.startLine ?? 0,
            endLine: m.endLine ?? 0,
            fileExtension: m.fileExtension || '',
            metadata
        };
    }

    async search(collectionName: string, queryVector: number[], options?: SearchOptions): Promise<VectorSearchResult[]> {
        await this.ensureInitialized();

        const topK = options?.topK || 10;

        try {
            const collectionId = await this.getCollectionId(collectionName);

            const body: any = {
                query_embeddings: [queryVector],
                n_results: topK,
                include: ['documents', 'metadatas', 'distances']
            };

            const where = this.buildWhere(options?.filterExpr, options?.filter);
            if (where) body.where = where;

            const response = await this.makeRequest(`/collections/${collectionId}/query`, 'POST', body);

            // Chroma returns per-query arrays; we only issue a single query vector.
            const ids: string[] = response?.ids?.[0] || [];
            const documents: string[] = response?.documents?.[0] || [];
            const metadatas: any[] = response?.metadatas?.[0] || [];
            const distances: number[] = response?.distances?.[0] || [];

            const results: VectorSearchResult[] = ids.map((id, i) => {
                const distance = distances[i];
                // With cosine space, distance = 1 - similarity. Convert back to a
                // similarity score so higher is better (matching Milvus COSINE).
                const score = typeof distance === 'number' ? 1 - distance : 0;
                return {
                    document: this.fromChromaRecord(id, documents[i], metadatas[i], queryVector),
                    score
                };
            });

            const threshold = options?.threshold;
            return threshold !== undefined ? results.filter(r => r.score >= threshold) : results;
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to search in collection '${collectionName}':`, error);
            throw error;
        }
    }

    async hybridSearch(collectionName: string, searchRequests: HybridSearchRequest[], options?: HybridSearchOptions): Promise<HybridSearchResult[]> {
        // ChromaDB has no native hybrid search; fall back to dense search using
        // the dense request (the one whose `data` is a numeric vector).
        console.warn('[ChromaRestfulDB] ⚠️  hybridSearch falling back to dense vector search (ChromaDB has no native BM25 hybrid).');

        const denseRequest = searchRequests.find(r => Array.isArray(r.data)) || searchRequests[0];
        const queryVector = (Array.isArray(denseRequest?.data) ? denseRequest.data : []) as number[];
        const limit = options?.limit || denseRequest?.limit || 10;

        const results = await this.search(collectionName, queryVector, {
            topK: limit,
            filterExpr: options?.filterExpr
        });

        return results.map(r => ({ document: r.document, score: r.score }));
    }

    async delete(collectionName: string, ids: string[]): Promise<void> {
        await this.ensureInitialized();

        if (ids.length === 0) return;

        try {
            const collectionId = await this.getCollectionId(collectionName);
            await this.makeRequest(`/collections/${collectionId}/delete`, 'POST', { ids });
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to delete documents from collection '${collectionName}':`, error);
            throw error;
        }
    }

    async query(collectionName: string, filter: string, outputFields: string[], limit?: number): Promise<Record<string, any>[]> {
        await this.ensureInitialized();

        try {
            // count(*) is handled specially via the dedicated count endpoint.
            if (outputFields.length === 1 && /count\(\*\)/i.test(outputFields[0])) {
                const count = await this.getCollectionRowCount(collectionName);
                return [{ 'count(*)': count < 0 ? 0 : count }];
            }

            const collectionId = await this.getCollectionId(collectionName);

            const body: any = {
                include: ['documents', 'metadatas']
            };
            const where = this.buildWhere(filter);
            if (where) body.where = where;
            if (limit !== undefined) body.limit = limit;

            const response = await this.makeRequest(`/collections/${collectionId}/get`, 'POST', body);

            const ids: string[] = response?.ids || [];
            const documents: string[] = response?.documents || [];
            const metadatas: any[] = response?.metadatas || [];

            // Project onto the requested output fields.
            return ids.map((id, i) => {
                const doc = this.fromChromaRecord(id, documents[i], metadatas[i], []);
                const record: Record<string, any> = {};
                for (const field of outputFields) {
                    switch (field) {
                        case 'id': record.id = doc.id; break;
                        case 'content': record.content = doc.content; break;
                        case 'relativePath': record.relativePath = doc.relativePath; break;
                        case 'startLine': record.startLine = doc.startLine; break;
                        case 'endLine': record.endLine = doc.endLine; break;
                        case 'fileExtension': record.fileExtension = doc.fileExtension; break;
                        case 'metadata': record.metadata = JSON.stringify(doc.metadata); break;
                        default: (record as any)[field] = (doc as any)[field]; break;
                    }
                }
                if (outputFields.length === 0) record.id = doc.id;
                return record;
            });
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to query collection '${collectionName}':`, error);
            throw error;
        }
    }

    async getCollectionDescription(collectionName: string): Promise<string> {
        await this.ensureInitialized();

        try {
            const response = await this.makeRequest(`/collections/${encodeURIComponent(collectionName)}`, 'GET');
            return response?.metadata?.description || '';
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Failed to get description for collection '${collectionName}':`, error);
            throw error;
        }
    }

    async checkCollectionLimit(): Promise<boolean> {
        // ChromaDB (self-hosted) has no per-account collection limit.
        return true;
    }

    async getCollectionRowCount(collectionName: string): Promise<number> {
        await this.ensureInitialized();

        try {
            const collectionId = await this.getCollectionId(collectionName);
            const response = await this.makeRequest(`/collections/${collectionId}/count`, 'GET');
            // The count endpoint returns a bare number.
            const n = typeof response === 'number' ? response : parseInt(String(response?.count ?? response), 10);
            return Number.isFinite(n) && n >= 0 ? n : -1;
        } catch (error) {
            console.error(`[ChromaRestfulDB] ❌ Error getting row count for '${collectionName}':`, error);
            return -1;
        }
    }

    /**
     * Best-effort translation of the Milvus-style boolean filter expressions used
     * by this codebase into a Chroma `where` object.
     *
     * Supported patterns (which cover the expressions produced internally):
     *   - `field == "value"`            -> { field: { "$eq": value } }
     *   - `field in ["a", "b"]`         -> { field: { "$in": [a, b] } }
     *   - multiple clauses joined by `and` / `&&` -> { "$and": [...] }
     *
     * An explicit `whereObj` (Record filter) takes precedence when provided.
     * Returns undefined when there is no usable filter.
     */
    private buildWhere(filterExpr?: string, whereObj?: Record<string, any>): Record<string, any> | undefined {
        if (whereObj && Object.keys(whereObj).length > 0) {
            return whereObj;
        }
        if (!filterExpr || filterExpr.trim().length === 0) {
            return undefined;
        }

        const clauses = filterExpr.split(/\s+(?:and|&&)\s+/i).map(c => c.trim()).filter(Boolean);
        const parsed: Record<string, any>[] = [];

        for (const clause of clauses) {
            // field in ["a", "b"]
            const inMatch = clause.match(/^(\w+)\s+in\s+\[(.*)\]$/i);
            if (inMatch) {
                const field = inMatch[1];
                const values = inMatch[2]
                    .split(',')
                    .map(v => this.parseScalar(v.trim()))
                    .filter(v => v !== undefined);
                if (values.length > 0) parsed.push({ [field]: { '$in': values } });
                continue;
            }

            // field == value  (also tolerate a single '=')
            const eqMatch = clause.match(/^(\w+)\s*==?\s*(.+)$/);
            if (eqMatch) {
                const field = eqMatch[1];
                const value = this.parseScalar(eqMatch[2].trim());
                if (value !== undefined) parsed.push({ [field]: { '$eq': value } });
                continue;
            }

            console.warn(`[ChromaRestfulDB] ⚠️  Unsupported filter clause ignored: "${clause}"`);
        }

        if (parsed.length === 0) return undefined;
        if (parsed.length === 1) return parsed[0];
        return { '$and': parsed };
    }

    /** Parse a scalar literal from a filter expression (string/number/bool). */
    private parseScalar(raw: string): any {
        if (raw.length === 0) return undefined;
        // Quoted string
        const quoted = raw.match(/^["'](.*)["']$/s);
        if (quoted) {
            // Undo the backslash escaping applied for Milvus expressions.
            return quoted[1].replace(/\\\\/g, '\\').replace(/\\"/g, '"');
        }
        if (/^-?\d+$/.test(raw)) return parseInt(raw, 10);
        if (/^-?\d*\.\d+$/.test(raw)) return parseFloat(raw);
        if (raw === 'true') return true;
        if (raw === 'false') return false;
        return raw;
    }
}
