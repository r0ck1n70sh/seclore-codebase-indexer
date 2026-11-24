#!/usr/bin/env node

// CRITICAL: Redirect console outputs to stderr IMMEDIATELY to avoid interfering with MCP JSON protocol
// Only MCP protocol messages should go to stdout
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;

console.log = (...args: any[]) => {
    process.stderr.write('[LOG] ' + args.join(' ') + '\n');
};

console.warn = (...args: any[]) => {
    process.stderr.write('[WARN] ' + args.join(' ') + '\n');
};

// console.error already goes to stderr by default

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
    ListToolsRequestSchema,
    CallToolRequestSchema,
    isInitializeRequest
} from "@modelcontextprotocol/sdk/types.js";
import * as http from "node:http";
import { randomUUID } from "node:crypto";
import { Context } from "@zilliz/claude-context-core";

// Import our modular components
import { createMcpConfig, logConfigurationSummary, showHelpMessage, ContextMcpConfig } from "./config.js";
import { createEmbeddingInstance, logEmbeddingProviderInfo } from "./embedding.js";
import { createVectorDatabaseInstance } from "./vectordb.js";
import { SnapshotManager } from "./snapshot.js";
import { SyncManager } from "./sync.js";
import { ToolHandlers } from "./handlers.js";

class ContextMcpServer {
    private config: ContextMcpConfig;
    private context: Context;
    private snapshotManager: SnapshotManager;
    private syncManager: SyncManager;
    private toolHandlers: ToolHandlers;

    constructor(config: ContextMcpConfig) {
        this.config = config;

        // Initialize embedding provider
        console.log(`[EMBEDDING] Initializing embedding provider: ${config.embeddingProvider}`);
        console.log(`[EMBEDDING] Using model: ${config.embeddingModel}`);

        const embedding = createEmbeddingInstance(config);
        logEmbeddingProviderInfo(config, embedding);

        // Initialize vector database (Milvus or Chroma, based on config)
        const vectorDatabase = createVectorDatabaseInstance(config);

        // Initialize Claude Context
        this.context = new Context({
            embedding,
            vectorDatabase,
            collectionNameOverride: config.collectionNameOverride
        });

        // Initialize managers
        this.snapshotManager = new SnapshotManager();
        this.syncManager = new SyncManager(this.context, this.snapshotManager);
        this.toolHandlers = new ToolHandlers(this.context, this.snapshotManager);

        // Load existing codebase snapshot on startup
        this.snapshotManager.loadCodebaseSnapshot();
    }

    /**
     * Create a fresh MCP Server instance with all tools registered. The heavy
     * shared resources (Context, embedding, vector DB, managers) are owned by
     * this class, so every Server created here operates on the same index and
     * sync state. Used once for stdio, or once per client session over HTTP.
     */
    private createServerInstance(): Server {
        const server = new Server(
            {
                name: this.config.name,
                version: this.config.version
            },
            {
                capabilities: {
                    tools: {}
                }
            }
        );
        this.setupTools(server);
        return server;
    }

    private setupTools(server: Server) {
        const index_description = `
Index a codebase directory to enable semantic search using a configurable code splitter.

⚠️ **IMPORTANT**:
- You MUST provide an absolute path to the target codebase.

✨ **Usage Guidance**:
- This tool is typically used when search fails due to an unindexed codebase.
- If indexing is attempted on an already indexed path, and a conflict is detected, you MUST prompt the user to confirm whether to proceed with a force index (i.e., re-indexing and overwriting the previous index).
`;


        const search_description = `
Search the indexed codebase using natural language queries within a specified absolute path.

⚠️ **IMPORTANT**:
- You MUST provide an absolute path.

🎯 **When to Use**:
This tool is versatile and can be used before completing various tasks to retrieve relevant context:
- **Code search**: Find specific functions, classes, or implementations
- **Context-aware assistance**: Gather relevant code context before making changes
- **Issue identification**: Locate problematic code sections or bugs
- **Code review**: Understand existing implementations and patterns
- **Refactoring**: Find all related code pieces that need to be updated
- **Feature development**: Understand existing architecture and similar implementations
- **Duplicate detection**: Identify redundant or duplicated code patterns across the codebase

✨ **Usage Guidance**:
- If the codebase is not indexed, this tool will return a clear error message indicating that indexing is required first.
- You can then use the index_codebase tool to index the codebase before searching again.
`;

        // Define available tools
        server.setRequestHandler(ListToolsRequestSchema, async () => {
            return {
                tools: [
                    {
                        name: "index_codebase",
                        description: index_description,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to index.`
                                },
                                force: {
                                    type: "boolean",
                                    description: "Force re-indexing even if already indexed",
                                    default: false
                                },
                                splitter: {
                                    type: "string",
                                    description: "Code splitter to use: 'ast' for syntax-aware splitting with automatic fallback, 'langchain' for character-based splitting",
                                    enum: ["ast", "langchain"],
                                    default: "ast"
                                },
                                customExtensions: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: Additional file extensions to include beyond defaults (e.g., ['.vue', '.svelte', '.astro']). Extensions should include the dot prefix or will be automatically added",
                                    default: []
                                },
                                ignorePatterns: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: Additional ignore patterns to exclude specific files/directories beyond defaults. Only include this parameter if the user explicitly requests custom ignore patterns (e.g., ['static/**', '*.tmp', 'private/**'])",
                                    default: []
                                }
                            },
                            required: ["path"]
                        }
                    },
                    {
                        name: "search_code",
                        description: search_description,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to search in.`
                                },
                                query: {
                                    type: "string",
                                    description: "Natural language query to search for in the codebase"
                                },
                                limit: {
                                    type: "number",
                                    description: "Maximum number of results to return",
                                    default: 10,
                                    maximum: 50
                                },
                                extensionFilter: {
                                    type: "array",
                                    items: {
                                        type: "string"
                                    },
                                    description: "Optional: List of file extensions to filter results. (e.g., ['.ts','.py']).",
                                    default: []
                                }
                            },
                            required: ["path", "query"]
                        }
                    },
                    {
                        name: "clear_index",
                        description: `Clear the search index. IMPORTANT: You MUST provide an absolute path.`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to clear.`
                                }
                            },
                            required: ["path"]
                        }
                    },
                    {
                        name: "get_indexing_status",
                        description: `Get the current indexing status of a codebase. Shows progress percentage for actively indexing codebases and completion status for indexed codebases.`,
                        inputSchema: {
                            type: "object",
                            properties: {
                                path: {
                                    type: "string",
                                    description: `ABSOLUTE path to the codebase directory to check status for.`
                                }
                            },
                            required: ["path"]
                        }
                    },
                ]
            };
        });

        // Handle tool execution
        server.setRequestHandler(CallToolRequestSchema, async (request) => {
            const { name, arguments: args } = request.params;

            switch (name) {
                case "index_codebase":
                    return await this.toolHandlers.handleIndexCodebase(args);
                case "search_code":
                    return await this.toolHandlers.handleSearchCode(args);
                case "clear_index":
                    return await this.toolHandlers.handleClearIndex(args);
                case "get_indexing_status":
                    return await this.toolHandlers.handleGetIndexingStatus(args);

                default:
                    throw new Error(`Unknown tool: ${name}`);
            }
        });
    }

    async start() {
        console.log('[SYNC-DEBUG] MCP server start() method called');
        console.log('Starting Context MCP server...');

        // One-shot startup healing for legacy 0/0+completed snapshot entries
        // left over from pre-fix MCP versions. Runs before the transport accepts
        // requests so clients never observe the poisoning state. See Issue #295.
        await this.toolHandlers.validateLegacyZeroEntries();

        if (this.config.transport === 'http') {
            await this.startHttp();
        } else {
            await this.startStdio();
        }

        // Start background sync after the transport is ready. The shared
        // Context/managers are used by every client session, so sync runs once.
        console.log('[SYNC-DEBUG] Initializing background sync...');
        this.syncManager.startBackgroundSync();
        console.log('[SYNC-DEBUG] MCP server initialization complete');
    }

    private async startStdio() {
        const transport = new StdioServerTransport();
        console.log('[SYNC-DEBUG] StdioServerTransport created, attempting server connection...');

        const server = this.createServerInstance();
        await server.connect(transport);
        console.log("MCP server started and listening on stdio.");
        console.log('[SYNC-DEBUG] Server connection established successfully');
    }

    /**
     * Start an HTTP server that speaks the MCP Streamable HTTP transport and
     * supports many concurrent client sessions. Each MCP session gets its own
     * transport + Server instance (keyed by the `mcp-session-id` header), while
     * all sessions share this class's Context/index/sync state.
     */
    private async startHttp() {
        // Active sessions keyed by MCP session id.
        const transports: Record<string, StreamableHTTPServerTransport> = {};

        const readBody = (req: http.IncomingMessage): Promise<any> =>
            new Promise((resolve, reject) => {
                const chunks: Buffer[] = [];
                req.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
                req.on('end', () => {
                    const raw = Buffer.concat(chunks).toString('utf8');
                    if (!raw) {
                        resolve(undefined);
                        return;
                    }
                    try {
                        resolve(JSON.parse(raw));
                    } catch (err) {
                        reject(err);
                    }
                });
                req.on('error', reject);
            });

        const sendJsonError = (res: http.ServerResponse, status: number, message: string, id: any = null) => {
            res.writeHead(status, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                jsonrpc: '2.0',
                error: { code: -32000, message },
                id
            }));
        };

        const httpServer = http.createServer(async (req, res) => {
            try {
                const url = new URL(req.url || '/', `http://${req.headers.host}`);

                // Lightweight liveness endpoint for load balancers / health checks.
                if (url.pathname === '/health') {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({
                        status: 'ok',
                        mode: this.config.httpStateless ? 'stateless' : 'stateful',
                        sessions: Object.keys(transports).length
                    }));
                    return;
                }

                if (url.pathname !== '/mcp') {
                    sendJsonError(res, 404, `Not found: ${url.pathname}`);
                    return;
                }

                if (this.config.httpStateless) {
                    // Stateless mode: no mcp-session-id validation. A brand-new
                    // transport + Server handles each request and is torn down when
                    // the response closes. This is restart-safe — there is no session
                    // for a client to lose or fail to "reconnect" to.
                    //
                    // All three MCP methods are routed through the transport so the
                    // SDK produces spec-correct responses: POST -> JSON/SSE reply,
                    // GET -> a 200 SSE stream (session validation is skipped when
                    // sessionIdGenerator is undefined), DELETE -> 200. GET is served
                    // as a real stream rather than 405 so clients that open a
                    // standalone SSE channel (e.g. Claude Code) connect cleanly.
                    if (req.method !== 'POST' && req.method !== 'GET' && req.method !== 'DELETE') {
                        sendJsonError(res, 405, 'Method not allowed');
                        return;
                    }

                    const body = req.method === 'POST' ? await readBody(req) : undefined;
                    const server = this.createServerInstance();
                    const transport = new StreamableHTTPServerTransport({
                        sessionIdGenerator: undefined
                    });
                    // Clean up the per-request transport/server once the client
                    // disconnects (immediately for POST/DELETE, when the SSE stream
                    // closes for GET).
                    res.on('close', () => {
                        transport.close().catch(() => { });
                        server.close().catch(() => { });
                    });
                    await server.connect(transport);
                    await transport.handleRequest(req, res, body);
                    return;
                }

                // Stateful mode: sessions keyed by the mcp-session-id header.
                const sessionId = req.headers['mcp-session-id'] as string | undefined;

                // GET (SSE stream) and DELETE (session teardown) require an
                // existing session; hand them straight to its transport. An
                // unknown/expired session returns 404 (per spec) so the client
                // discards it and re-initializes rather than failing hard.
                if (req.method === 'GET' || req.method === 'DELETE') {
                    const transport = sessionId ? transports[sessionId] : undefined;
                    if (!transport) {
                        sendJsonError(res, 404, 'Session not found or expired; re-initialize');
                        return;
                    }
                    await transport.handleRequest(req, res);
                    return;
                }

                if (req.method !== 'POST') {
                    sendJsonError(res, 405, 'Method not allowed');
                    return;
                }

                const body = await readBody(req);
                let transport = sessionId ? transports[sessionId] : undefined;

                if (!transport) {
                    // No session id supplied (or an unknown one). Only an
                    // `initialize` request may open a new session; anything else
                    // gets 404 so the client knows to re-initialize.
                    if (!isInitializeRequest(body)) {
                        sendJsonError(res, 404, 'Session not found or expired; re-initialize');
                        return;
                    }

                    transport = new StreamableHTTPServerTransport({
                        sessionIdGenerator: () => randomUUID(),
                        onsessioninitialized: (sid) => {
                            transports[sid] = transport!;
                            console.log(`[HTTP] Session initialized: ${sid} (active: ${Object.keys(transports).length})`);
                        }
                    });

                    transport.onclose = () => {
                        const sid = transport!.sessionId;
                        if (sid && transports[sid]) {
                            delete transports[sid];
                            console.log(`[HTTP] Session closed: ${sid} (active: ${Object.keys(transports).length})`);
                        }
                    };

                    // Each session drives its own Server instance over the shared state.
                    const server = this.createServerInstance();
                    await server.connect(transport);
                }

                await transport.handleRequest(req, res, body);
            } catch (error: any) {
                console.error(`[HTTP] Request handling error: ${error?.message || error}`);
                if (!res.headersSent) {
                    sendJsonError(res, 500, 'Internal server error');
                }
            }
        });

        await new Promise<void>((resolve) => {
            httpServer.listen(this.config.httpPort, this.config.httpHost, resolve);
        });

        const mode = this.config.httpStateless ? 'stateless' : 'stateful (session-based)';
        console.log(`MCP server started and listening on http://${this.config.httpHost}:${this.config.httpPort}/mcp [${mode}]`);
    }
}

// Main execution
async function main() {
    // Parse command line arguments
    const args = process.argv.slice(2);

    // Show help if requested
    if (args.includes('--help') || args.includes('-h')) {
        showHelpMessage();
        process.exit(0);
    }

    // Create configuration
    const config = createMcpConfig();
    logConfigurationSummary(config);

    const server = new ContextMcpServer(config);
    await server.start();
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.error("Received SIGINT, shutting down gracefully...");
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.error("Received SIGTERM, shutting down gracefully...");
    process.exit(0);
});

// Always start the server - this is designed to be the main entry point
main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});
