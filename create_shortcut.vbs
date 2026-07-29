' Creates a Desktop shortcut that runs CriGent from the Python source in this
' folder. Every path is derived from where this script lives, so the folder can
' be moved or renamed and re-running this still produces a correct shortcut.

Set oWS  = WScript.CreateObject("WScript.Shell")
Set oFSO = CreateObject("Scripting.FileSystemObject")

sHere    = oFSO.GetParentFolderName(WScript.ScriptFullName)
sDesktop = oWS.SpecialFolders("Desktop")
sTarget  = oFSO.BuildPath(sHere, "launch_crigent.bat")
sIcon    = oFSO.BuildPath(sHere, "crigent.ico")

If Not oFSO.FileExists(sTarget) Then
    WScript.Echo "Cannot find launch_crigent.bat next to this script (" & sHere & ")"
    WScript.Quit 1
End If

sLink = oFSO.BuildPath(sDesktop, "CriGent (source).lnk")
Set oLink = oWS.CreateShortcut(sLink)
oLink.TargetPath       = sTarget
oLink.WorkingDirectory = sHere
oLink.Description      = "CriGent - run from the Python source in " & sHere
oLink.WindowStyle      = 7            ' start minimised; the app opens its own window
If oFSO.FileExists(sIcon) Then
    oLink.IconLocation = sIcon & ",0"
End If
oLink.Save

WScript.Echo "Created: " & sLink
WScript.Echo "  runs : " & sTarget
