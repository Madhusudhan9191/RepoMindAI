import React, { useState } from "react";
import { Folder, FolderOpen, FileText } from "lucide-react";

function buildTree(files) {
  const root = { name: "Workspace Root", isDirectory: true, children: {} };
  
  files.forEach((file) => {
    const parts = file.path.split("/");
    let current = root;
    
    parts.forEach((part, index) => {
      const isLast = index === parts.length - 1;
      if (!current.children[part]) {
        current.children[part] = {
          name: part,
          path: file.path,
          isDirectory: !isLast,
          children: isLast ? null : {},
        };
      }
      current = current.children[part];
    });
  });
  
  return root;
}

function TreeNode({ node, depth }) {
  const [isOpen, setIsOpen] = useState(depth === 0);

  if (!node.isDirectory) {
    return (
      <div 
        className="tree-node file" 
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        title={node.path}
      >
        <FileText className="node-icon" size={14} />
        <span>{node.name}</span>
      </div>
    );
  }

  const sortedKeys = Object.keys(node.children || {}).sort((a, b) => {
    const aIsDir = node.children[a].isDirectory;
    const bIsDir = node.children[b].isDirectory;
    if (aIsDir && !bIsDir) return -1;
    if (!aIsDir && bIsDir) return 1;
    return a.localeCompare(b);
  });

  return (
    <div>
      <div 
        className="tree-node directory" 
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? (
          <FolderOpen className="node-icon" size={14} style={{ color: "#6366f1" }} />
        ) : (
          <Folder className="node-icon" size={14} style={{ color: "#818cf8" }} />
        )}
        <span style={{ fontWeight: 500, color: "#e5e7eb" }}>{node.name}</span>
      </div>
      {isOpen && (
        <div className="node-children">
          {sortedKeys.map((key) => (
            <TreeNode 
              key={node.children[key].path + "-" + key} 
              node={node.children[key]} 
              depth={depth + 1} 
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function TreeView({ files }) {
  if (!files || files.length === 0) {
    return (
      <div style={{ padding: "8px", color: "#6b7280", fontStyle: "italic", fontSize: "0.80rem" }}>
        No repository index files discovered.
      </div>
    );
  }

  const tree = buildTree(files);
  const topKeys = Object.keys(tree.children).sort((a, b) => {
    const aIsDir = tree.children[a].isDirectory;
    const bIsDir = tree.children[b].isDirectory;
    if (aIsDir && !bIsDir) return -1;
    if (!aIsDir && bIsDir) return 1;
    return a.localeCompare(b);
  });

  return (
    <div className="tree-container">
      {topKeys.map((key) => (
        <TreeNode 
          key={tree.children[key].path + "-" + key} 
          node={tree.children[key]} 
          depth={0} 
        />
      ))}
    </div>
  );
}
