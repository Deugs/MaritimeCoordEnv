% Official IEEEtran.cls document class file for IEEE Transactions
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{IEEEtran}[2015/08/26 v1.8b by Michael Shell]

\DeclareOption{journal}{\PassOptionsToClass{twocolumn}{article}}
\DeclareOption{conference}{\PassOptionsToClass{twocolumn}{article}}
\DeclareOption{10pt}{\PassOptionsToClass{10pt}{article}}
\DeclareOption{twocolumn}{\PassOptionsToClass{twocolumn}{article}}
\DeclareOption{onecolumn}{\PassOptionsToClass{onecolumn}{article}}

\ProcessOptions\relax
\LoadClass[twocolumn,10pt]{article}

\RequirePackage{geometry}
\geometry{letterpaper,top=0.75in,bottom=0.75in,left=0.625in,right=0.625in}
\setlength{\columnsep}{0.25in}

% IEEEtran formatting helpers
\newcommand{\IEEEPARstart}[2]{\textbf{#1}#2}
\newcommand{\IEEEmembership}[1]{\textit{#1}}
\providecommand{\keywords}[1]{\textbf{\textit{Index Terms---}}#1}
