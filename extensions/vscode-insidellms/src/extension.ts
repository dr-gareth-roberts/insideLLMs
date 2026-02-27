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

function workspaceRoot(): string | undefined {
  const folder = vscode.workspace.workspaceFolders?.[0];
  return folder?.uri.fsPath;
}

function commandLine(configPath: string, runDir: string): string {
  const quotedConfig = JSON.stringify(configPath);
  const quotedRunDir = JSON.stringify(runDir);
  return `insidellms harness ${quotedConfig} --run-dir ${quotedRunDir} --overwrite --skip-report`;
}

export function activate(context: vscode.ExtensionContext): void {
  const provider = new InsideLLMsCodeLensProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeLensProvider({ language: "python" }, provider)
  );

  const disposable = vscode.commands.registerCommand(
    RUN_PROBES_COMMAND,
    async (_fileUri?: string) => {
      const root = workspaceRoot();
      if (!root) {
        vscode.window.showErrorMessage(
          "insideLLMs: open a workspace folder to run harness probes."
        );
        return;
      }

      const cfg = vscode.workspace.getConfiguration("insidellms");
      const harnessConfigPath = cfg.get<string>("harnessConfigPath", "ci/harness.yaml");
      const runDir = cfg.get<string>("runDir", ".tmp/runs/ide");
      const terminal = vscode.window.createTerminal({
        name: "insideLLMs Probes",
        cwd: root
      });
      terminal.show(true);
      terminal.sendText(commandLine(path.join(root, harnessConfigPath), path.join(root, runDir)));
    }
  );
  context.subscriptions.push(disposable);
}

export function deactivate(): void {}
