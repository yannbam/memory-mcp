#!/bin/bash
# Simple script to verify both tool exposure modes work correctly

echo "Building project..."
npm run build

echo ""
echo "=========================================="
echo "Testing DEFAULT mode (single 'memory' tool)"
echo "=========================================="
echo ""
echo "Starting server in default mode..."
echo "Expected: Single 'memory' tool with command parameter"
echo ""
echo "Run this in another terminal:"
echo "  npx @modelcontextprotocol/inspector dist/index.js"
echo ""
echo "Or add to .mcp.json:"
echo '  "memory-mcp-default": {'
echo '    "command": "node",'
echo '    "args": ["'$(pwd)'/dist/index.js"]'
echo '  }'
echo ""
read -p "Press Enter when you've verified the default mode..."

echo ""
echo "=========================================="
echo "Testing ONE-TOOL-PER-COMMAND mode"
echo "=========================================="
echo ""
echo "Starting server with --one-tool-per-command flag..."
echo "Expected: 6 separate tools (memory_view, memory_create, memory_str_replace, memory_insert, memory_delete, memory_rename)"
echo ""
echo "Run this in another terminal:"
echo "  npx @modelcontextprotocol/inspector dist/index.js --one-tool-per-command"
echo ""
echo "Or add to .mcp.json:"
echo '  "memory-mcp-separate": {'
echo '    "command": "node",'
echo '    "args": ["'$(pwd)'/dist/index.js", "--one-tool-per-command"]'
echo '  }'
echo ""
read -p "Press Enter when you've verified the one-tool-per-command mode..."

echo ""
echo "=========================================="
echo "Testing complete!"
echo "=========================================="
echo ""
echo "Both modes should work with the same underlying operations."
echo "The only difference is how the tools are exposed to the client."
