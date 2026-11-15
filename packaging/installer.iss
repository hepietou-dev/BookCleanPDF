; BookClean PDF — Inno Setup 安装脚本（M5.5）
; 交付方式：PyInstaller onedir（dist/BookCleanPDF）。运行时 Python 已捆绑，用户无需另装 Python。
; 离线安装：不下载任何内容、不联网。
; 数据安全：卸载仅移除 {app} 内本脚本安装的文件；用户数据 %LOCALAPPDATA%\BookCleanPDF
;           与用户输出 PDF（文档\BookCleanPDF）绝不删除（[UninstallDelete] 故意留空）。

#define MyAppName "BookClean PDF"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "BookClean PDF Project"
#define MyAppExeName "BookCleanPDF.exe"

[Setup]
AppId={{7B9E4A2C-3D5F-4E6A-9B8C-1F2E3D4C5B6A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={userpf}\BookCleanPDF
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
OutputDir=..\dist\installer
OutputBaseFilename=BookCleanPDF-{#MyAppVersion}-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; _per-user_ 安装：无需管理员/UAC，无需命令行；双击即可离线安装。
PrivilegesRequired=lowest
VersionInfoVersion={#MyAppVersion}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "chs"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\BookCleanPDF\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
; 故意留空：不删除任何用户数据或输出 PDF。Inno 默认仅移除本脚本安装的文件与 {app} 目录。