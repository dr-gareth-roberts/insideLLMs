import * as path from "path";
import * as vscode from "vscode";

const RUN_PROBES_COMMAND = "insidellms.runProbes";

class InsideLLMsCodeLensProvider implements vscode.CodeLensProvider {
  provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
    const lenses: vscode.CodeLens[] = [];
    const fileUri = document.uri.toString();

    const topLens = new vscode.CodeLens(new vscode.Range(0, 0, 0, 0), {
      command: RUN_PROBES_COMMAND,
      title: "Run insideLLMs probes",
      arguments: [fileUri]
    });
    lenses.push(topLens);

    const promptRegex = /\b(prompt|system_prompt|assistant_prompt)\b\s*=/;
    for (let i = 0; i < document.lineCount; i += 1) {
      const text = document.lineAt(i).text;
      if (!promptRegex.test(text)) {
        continue;
      }
      lenses.push(
        new vscode.CodeLens(new vscode.Range(i, 0, i, 0), {
          command: RUN_PROBES_COMMAND,
          title: "Run insideLLMs probes",
          arguments: [fileUri]
        })
      );
    }
    return lenses;
  }
}

async function invokingWorkspace(fileUri?: string | vscode.Uri): Promise<vscode.WorkspaceFolder | undefined> {
  const uri = typeof fileUri === 'string' ? vscode.Uri.parse(fileUri) : fileUri;
  const documentUri = uri ?? vscode.window.activeTextEditor?.document.uri;
  if (documentUri) {
    return vscode.workspace.getWorkspaceFolder(documentUri);
  }
  const folders = vscode.workspace.workspaceFolders;
  return folders?.length === 1 ? folders[0] : vscode.window.showWorkspaceFolderPick();
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new InsideLLMsCodeLensProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider({ language: "python" }, provider)
  );

  const disposable = vscode.commands.registerCommand(
    RUN_PROBES_COMMAND,
    async (fileUri?: string | vscode.Uri) => {
      if (!vscode.workspace.isTrusted) {
        vscode.window.showErrorMessage('insideLLMs: trust this workspace before running probes.');
        return;
      }
      const folder = await invokingWorkspace(fileUri);
      if (!folder) {
        vscode.window.showErrorMessage(
          "insideLLMs: open a workspace folder to run harness probes."
        );
        return;
      }

      const root = folder.uri.fsPath;
      const cfg = vscode.workspace.getConfiguration('insidellms', folder.uri);
      const harnessConfigPath = cfg.get<string>("harnessConfigPath", "ci/harness.yaml");
      const runDir = cfg.get<string>("runDir", ".tmp/runs/ide");
      const execution = new vscode.ProcessExecution('insidellms', [
        'harness', path.resolve(root, harnessConfigPath),
        '--run-dir', path.resolve(root, runDir), '--overwrite', '--skip-report'
      ], { cwd: root });
      const task = new vscode.Task(
        { type: 'insidellms', workspace: folder.uri.toString() }, folder,
        'Run Probes', 'insideLLMs', execution
      );
      try {
        return await vscode.tasks.executeTask(task);
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : String(error);
        vscode.window.showErrorMessage(`insideLLMs: failed to launch probes: ${message}`);
      }
    }
  );
  context.subscriptions.push(disposable);
}

export function deactivate(): void {}
