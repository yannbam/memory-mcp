# Human Notes - DONT EDIT!!!

TODO: change the input schema to a discriminated union schema


The memory tool has 6 different command types. You need a discriminated union schema. Here are both implementations:

  TypeScript / Zod Implementation:

  import { z } from 'zod';

  // Define each command as a separate Zod schema
  const viewCommand = z.object({
    command: z.literal('view'),
    path: z.string(),
    view_range: z.array(z.number()).length(2).optional(),
  });

  const createCommand = z.object({
    command: z.literal('create'),
    path: z.string(),
    file_text: z.string(),
  });

  const strReplaceCommand = z.object({
    command: z.literal('str_replace'),
    path: z.string(),
    old_str: z.string(),
    new_str: z.string(),
  });

  const insertCommand = z.object({
    command: z.literal('insert'),
    path: z.string(),
    insert_line: z.number().int(),
    insert_text: z.string(),
  });

  const deleteCommand = z.object({
    command: z.literal('delete'),
    path: z.string(),
  });

  const renameCommand = z.object({
    command: z.literal('rename'),
    old_path: z.string(),
    new_path: z.string(),
  });

  // Discriminated union: Zod will use 'command' field to narrow the type
  const memoryCommandInput = z.discriminatedUnion('command', [
    viewCommand,
    createCommand,
    strReplaceCommand,
    insertCommand,
    deleteCommand,
    renameCommand,
  ]);

  // Register the tool
  server.registerTool(
    'memory',
    {
      description: 'Manage persistent memory files with view, create, edit, delete, and rename operations',
      inputSchema: memoryCommandInput,
    },
    async (input) => {
      // input is strongly typed and narrowed by command field
      switch (input.command) {
        case 'view':
          return await handleView(input);
        case 'create':
          return await handleCreate(input);
        // ... etc
      }
    }
  );


  ----

  And let's implement both approaches:
    a) one unified tool with command parameter
    b) one tool per command
    switchable with a server cli arg

  Also implement --include-commands and --exclude-commands for which tools/commands get exposed by the server