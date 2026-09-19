%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% friteza.cls, v1, 2026/05/21
%
% This class file enables authors to prepare bachelor's
% and master's thesis at the University of Ljubljana,
% Faculty of Computer and Information Science.
%
% This template was based on pnas-new.cls, v1.2,
% 2016/02/28, which is included in a separate file.
%
% Modifications were made by
%     Iztok Lebar Bajec
%     Lidija Stanovnik
%     Matej Kristan
%     Žiga Emeršič

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either
% version 1.3 of this license or any later version.
% The latest version of this license is in
% http://www.latex-project.org/lppl.txt and
% version 1.3 or later is part of all distributions
% of LaTeX version 2005/12/01 or later.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% For use with latex+dvipdfm, pdflatex, xelatex & lualatex
% For compiling with plain latex, please use latex+dvipdfm
% to produce the PDF, not dvis -> ps -> pdf nor dvipdf
%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{style/friteza}[29/05/2026, v1]
\AtEndOfClass{\RequirePackage[nopatch=eqnum]{microtype}}



% Avoid object/xref streams if your engine/distribution enables them.
% Must be before PDF output starts.
\ifdefined\pdfobjcompresslevel
  \pdfobjcompresslevel=0
\fi
\ifdefined\pdfvariable
  \pdfvariable objcompresslevel=0
\fi

% Option for line numbers
\newif\if@pnaslineno
\DeclareOption{lineno}{\@pnaslinenotrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{extarticle}}
\ProcessOptions*
\LoadClass{extarticle}

% option for EMAI
\newif\if@emai

%% Fonts and language
\RequirePackage[utf8]{inputenc}
\RequirePackage[english,slovene]{babel}
\RequirePackage{amsmath,amsfonts,amssymb}
\RequirePackage{lmodern}
\RequirePackage[scaled]{helvet}
\RequirePackage[T1]{fontenc}
\RequirePackage{lettrine} % For dropped capitals

%% For the Significance Statement & footnote on the first page
\RequirePackage{afterpage}
\RequirePackage{ifpdf,ifxetex}
\ifpdf\else
  \ifxetex\else
    \def\pgfsysdriver{pgfsys-dvipdfm.def}
    \pdfpagewidth=\paperwidth
    \pdfpageheight=\paperheight
\fi\fi
\RequirePackage{xcolor}
\RequirePackage{tikz}
\RequirePackage[framemethod=tikz]{mdframed}

%% For single column equations and balancing the columns on final page
\RequirePackage{style/widetext}

\RequirePackage[explicit]{titlesec}
%% Hyperlinking
\RequirePackage{hyperref}
\hypersetup{colorlinks=true,allcolors=blue}

\makeatletter
\renewcommand{\title}[2][]{
  \ifstrequal{#1}{slovene}{
    \gdef\@title@sl{#2}
  }{
    \ifstrequal{#1}{english}{
        \gdef\@title@en{#2}
    }{
    \ClassError{friteza}{Invalid language option #1}{Select one of the supported languages for the title: slovene or english.}
    }
  }
}
\makeatother

%% Set up main title page fonts
\newcommand{\headerfont}{\normalfont\sffamily\fontsize{7}{9} \selectfont}
\newcommand{\footerfont}{\normalfont\sffamily\fontsize{7}{9} \selectfont}
%\newcommand{\titlefont}{\fontfamily{lmss}\bfseries\fontsize{22pt}{24pt}\selectfont}
\newcommand{\titlefont}{\normalfont\sffamily\bfseries\fontsize{22pt}{24pt}\selectfont}
%\newcommand{\dropcapfont}{\fontfamily{lmss}\bfseries\fontsize{26pt}{28pt}\selectfont}
\newcommand{\dropcapfont}{\normalfont\sffamily\bfseries\fontsize{26pt}{28pt}\selectfont}
\newcommand{\datesfont}{\normalfont\sffamily\fontsize{7}{8}\selectfont}
\newcommand{\absfont}{\normalfont\sffamily\bfseries\fontsize{8}{11}\selectfont}
\newcommand{\keywordsfont}{\normalfont\sffamily\fontsize{7}{10}\selectfont}
\newcommand{\copyrightfont}{\normalfont\rmfamily\fontsize{6}{8}\selectfont}

%% Set URL link color & font
\renewcommand\UrlFont{\color{black}\sffamily}

%% Author and affiliation
\RequirePackage{authblk}
\setlength{\affilsep}{8.5pt} % 16.5pts between base of author line and base of affil line
\renewcommand\Authfont{\color{color0}\normalfont\sffamily\bfseries\fontsize{9}{11}\selectfont}
\renewcommand\Affilfont{\color{color0}\normalfont\sffamily\fontsize{7}{8}\selectfont}
\makeatletter
\renewcommand\AB@affilsepx{; \protect\Affilfont}
\makeatother
\renewcommand\Authands{, and }

\RequirePackage{silence}
\WarningFilter{latex}
  {You have requested package `atbegshi'}

\WarningFilter{latex}
  {You have requested package `atveryend'}

\WarningFilter*{catoptions}
  {%
    \MessageBreak catoptions modifies package option handling and is%
    \MessageBreak incompatible with many other packages%
  }

%% Choose template type
\newcommand*{\templatetype}[1]{%
  \RequirePackage{#1}}

%% Options for element switching
\RequirePackage{xifthen}
\newboolean{shortarticle}
\newboolean{singlecolumn}

%% For numbering just one line of an equation
\newcommand\numberthis{\addtocounter{equation}{1}\tag{\theequation}}

%% Watermark
% \usepackage[printwatermark]{xwatermark}
\newboolean{displaywatermark}
\setboolean{displaywatermark}{false} % Set to false to remove the watermark
\ifthenelse{\boolean{displaywatermark}}{%
\newwatermark[allpages,color=gray!20,angle=45,scale=3,xpos=0,ypos=0]{DRAFT}}{}

%% Copyright statement (not used)
\newboolean{displaycopyright}
\setboolean{displaycopyright}{false} % Confirmed as not required
\RequirePackage{textcomp} % For copyright symbol styling
\newcommand{\copyrightstatement}{\, \textcopyright\, 2015 by The National Academy of Sciences of the USA}

%% Graphics, tables and other formatting
\RequirePackage{graphicx,xcolor}
\RequirePackage{colortbl}
\RequirePackage{booktabs}
\RequirePackage{algorithm}
\RequirePackage[noend]{algpseudocode}
\RequirePackage{changepage}
\RequirePackage[asymmetric,%
				a4paper,includeheadfoot,%
				%layoutsize={8.125in,10.875in},%
                %layouthoffset=0.1875in,%
                %layoutvoffset=0.0625in,%
                left=2.25in+38.5pt,%
                right=43pt,%
                top=43pt,% 10pt provided by headsep
                bottom=32pt,%
                headheight=0pt,% No Header
                headsep=10pt,%
                footskip=25pt]{geometry}
% \RequirePackage[labelfont={bf,sf},%
%                 labelsep=period,%
%                 figurename={\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Slika}{Figure}},justification=RaggedRight,singlelinecheck=false]{caption}
\RequirePackage{caption}
\captionsetup[figure]{labelfont={bf,sf},labelsep=period,name={\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Slika}{Figure}},justification=RaggedRight,singlelinecheck=false}
\captionsetup[algorithm]{labelfont={bf,sf},labelsep=period,name={\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Algoritem}{Algorithm}},justification=RaggedRight,singlelinecheck=false}
\captionsetup[listing]{labelfont={bf,sf},labelsep=period,name={\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Izsek kode}{Listing}},justification=RaggedRight,singlelinecheck=false}
\RequirePackage{multicol}
\setlength{\columnsep}{13.5pt} % Distance between the two columns of text
\setlength{\parindent}{12pt} % Paragraph indent
\RequirePackage{ragged2e}
\setlength{\RaggedRightParindent}{\parindent}

%% Set document color scheme
\definecolor{black50}{gray}{0.5} % 50% black for hrules
\definecolor{color0}{RGB}{0,0,0} % Base
\definecolor{color1}{RGB}{59,90,198} % author email, doi
\definecolor{color2}{RGB}{16,131,16} %
% For sig statement box
\definecolor{pnasbluetext}{RGB}{0,101,165} %
\definecolor{pnasblueback}{RGB}{205,217,235} %
%\definecolor{pnasbluetext}{RGB}{0,115,209} % Not used
%\definecolor{pnasblueback}{RGB}{210,230,247} % Not used
\definecolor{P1797}{RGB}{240,14,14}
\definecolor{black80}{RGB}{51,51,51}

%% Bibliography
% \RequirePackage[super,comma,sort&compress]{natbib}
\RequirePackage[numbers,sort&compress]{natbib}
\setlength{\bibsep}{0.0pt}
% You can use style 'abbrv' for sorting references by authors.
\bibliographystyle{style/pnas2011}
\renewcommand{\bibsection}{
  \expandafter\ifstrequal\expandafter{\languagename}{slovene}
    {\section*{Literatura}}
    {\section*{Bibliography}}
} % Remove header
%\renewcommand{\bibsection}{\noindent{\large\sffamily\bfseries \expandafter\ifstrequal\expandafter{\languagename}{slovene}{Literatura}{Bibliography}}} % Remove header
\renewcommand\bibfont{\normalfont\sffamily\fontsize{6}{8}\selectfont} % set font to be sans serif

\makeatletter
\renewcommand\@biblabel[1]{ #1.} % Remove brackets from label
\def\tagform@#1{\maketag@@@{\bfseries(\ignorespaces#1\unskip\@@italiccorr)}}
\renewcommand{\eqref}[1]{\textup{{\normalfont (\ref{#1}}\normalfont)}}
\makeatother

\makeatletter
%% Figure caption style
\DeclareCaptionFormat{pnasformat}{\normalfont\sffamily\fontsize{7}{9}\selectfont#1#2#3}
\captionsetup{format=pnasformat}

%% Table style
\RequirePackage{etoolbox}
%\captionsetup[table]{labelfont+={small},textfont+={small,sf,bf},skip=10pt,position=above}

\captionsetup[table]{labelfont={bf,sf},labelsep=period,skip=10pt,position=above}
% booktabs provide nice spacing, but rule widths and distances need fixing
\setlength{\heavyrulewidth}{0.5pt}
\setlength{\lightrulewidth}{0.5pt}
\setlength{\aboverulesep}{1.5pt}
\setlength{\belowrulesep}{1.5pt}
\setlength{\belowbottomsep}{10pt}
\AtBeginEnvironment{tabular}{
\sffamily\fontsize{7.5}{10}\selectfont
}
\newcommand{\addtabletext}[1]{{\setlength{\leftskip}{9pt}\fontsize{7}{9}\selectfont#1}}
\makeatother

%% Equation numbering - use square brackets
\makeatletter
\renewcommand\tagform@[1]{\maketag@@@ {(\ignorespaces #1\unskip \@@italiccorr )}}
\makeatother

%% Headers and footers
\RequirePackage{fancyhdr}  % custom headers/footers
\RequirePackage{lastpage}  % Number of pages in the document
\pagestyle{fancy}          % Enables the custom headers/footers

%% For the line numbers overlay
\def\leftlinenos{%
  \pgfmathtruncatemacro{\leftstartlineno}{2*(\thepage - 1)*62 + 1}%
  \pgfmathtruncatemacro{\leftendlineno}{(2*(\thepage - 1) + 1)*62}%
  \foreach \x in {\leftstartlineno,...,\leftendlineno}{\noindent\x\\}%
}
\def\rightlinenos{%
  \pgfmathtruncatemacro{\rightstartlineno}{(2*(\thepage - 1) + 1)*62 + 1}%
  \pgfmathtruncatemacro{\rightendlineno}{(2*\thepage)*62}%
  \foreach \x in {\rightstartlineno,...,\rightendlineno}{\noindent\x\\}%
}

\makeatletter
\fancypagestyle{firststyle}{
   \fancyfoot[R]{\ifboolexpr{test {\ifundef{\teza@program}} or test {\ifundef{\teza@vol}}}{}{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\ifbool{@diploma}{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{diplomsko delo}{Bachelor's thesis}}{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{magistrsko delo}{Master's thesis}}\hspace{7pt}%
   |\hspace{7pt}\teza@vol\hspace{7pt}|\hspace{7pt}\textbf{\thepage\textendash\pageref{LastPage}}}}

%   \fancyfoot[R]{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\textbf{FRI}teza\hspace{7pt}|\hspace{7pt}vol. \teza@vol\hspace{7pt}|\hspace{7pt}no. \teza@no\hspace{7pt}|\hspace{7pt}\textbf{\thepage\textendash\pageref{LastPage}}}
%   \fancyfoot[R]{\ifboolexpr{test {\ifundef{\teza@program}} or test {\ifundef{\teza@no}} or test {\ifundef{\teza@vol}}}{}{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\color{P1797}\textbf{FRI}\color{color0}teza\hspace{7pt}|\hspace{7pt}\teza@vol\hspace{7pt}|\hspace{7pt}\teza@no\hspace{7pt}|\hspace{7pt}\textbf{\thepage\textendash\pageref{LastPage}}}}
   %\fancyfoot[L]{\hspace*{-2.25in}\footerfont\@ifundefined{@doi}{}{\@doi}}
   \fancyfoot[L]{\hspace*{-2.25in}\footerfont}
}
\makeatother

% Headers
\fancyhead[LE,RO]{}
\fancyhead[LO,RE]{}
% Footers
\lfoot{}%
\cfoot{}%
\rfoot{}%
\makeatletter
%\fancyfoot[LE]{\hspace*{-2.25in}\footerfont\textbf{\thepage}\hspace{7pt}|\hspace{7pt}\@ifundefined{@doi}{}{\@doi}}
\fancyfoot[LE]{\hspace*{-2.25in}\footerfont\textbf{\thepage}\hspace{7pt}}
\fancyfoot[RO]{\ifboolexpr{test {\ifundef{\teza@program}} or test {\ifundef{\teza@vol}}}{}{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\ifbool{@diploma}{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{diplomsko delo}{Bachelor's thesis}}{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{magistrsko delo}{Master's thesis}}\hspace{7pt}|\hspace{7pt}\teza@vol\hspace{7pt}|\hspace{7pt}\textbf{\thepage}}}
%\fancyfoot[RO]{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\textbf{FRI}teza\hspace{7pt}|\hspace{7pt}vol. \teza@vol\hspace{7pt}|\hspace{7pt}no. \tezaatno\hspace{7pt}|\hspace{7pt}\textbf{\thepage}}
%\fancyfoot[RO]{\ifboolexpr{test {\ifundef{\teza@program}} or test {\ifundef{\teza@no}} or test {\ifundef{\teza@vol}}}{}{\footerfont \teza@program\hspace{7pt}|\hspace{7pt}\color{P1797}\textbf{FRI}\color{color0}teza\hspace{7pt}|\hspace{7pt}\teza@vol\hspace{7pt}|\hspace{7pt}\teza@no\hspace{7pt}|\hspace{7pt}\textbf{\thepage}}}
%\fancyfoot[RE,LO]{\hspace*{-2.25in}\footerfont\@ifundefined{@leadauthor}{}{\@leadauthor}\ifnum \value{authors} > 1\hspace{5pt}\emph{\@ifundefined{@etal}{}{\@etal}}\fi}
\fancyfoot[RE,LO]{\hspace*{-2.25in}\footerfont\@ifundefined{@leadauthor}{}{\color{black80}\emph{\@leadauthor}}}


% Use footer routine for line numbers
\AtBeginDocument{
  % first page of hte emai will not be used
%  \if@emai
%    \clearpage
%    \thispagestyle{empty}%
%    \begin{tikzpicture}[remember picture,overlay,inner sep=0pt,outer sep=0pt]
%    \node[anchor=center] at ([yshift=.3\paperheight]current page.south) {%
%      \includegraphics[width=\paperwidth]{logo/EMAI_header.jpg}%
%    };
%    \end{tikzpicture}%
%    \clearpage
%  \fi
  \RaggedRight
  \if@pnaslineno
    \ifthenelse{\boolean{singlecolumn}}{
      % use lineno package if singlecolumn
      \RequirePackage{lineno}
      \linenumbers
    }{% use tikz if twocolumn
    \fancyfoot[C]{\begin{tikzpicture}[remember picture,overlay]
    \node at([xshift=1.5em,yshift=\dimexpr -0.0625in-53pt] current page.north west)[anchor=north west,text width=3em,font=\rmfamily,align=right] {\leftlinenos};%
    \node at([xshift=-1.5em,yshift=\dimexpr -0.0625in-53pt] current page.north east)[anchor=north east,text width=3em,font=\rmfamily,align=left] {\rightlinenos};%
    \end{tikzpicture}}
    }
  \fi
}
\makeatother

\renewcommand{\headrulewidth}{0pt}% % No header rule
\renewcommand{\footrulewidth}{0pt}% % No footer rule

%% Section/subsection/paragraph set-up
\makeatletter

\let\oldsection\section
\RenewDocumentCommand{\section}{s o m}{%
  \IfBooleanTF{#1}{%
    \phantomsection%
    \oldsection*{#3}% \section*{..}
    \NR@gettitle{#3}% enable \nameref
    \addcontentsline{toc}{section}{#3}% starred sections are listed manually
  }{%
    \IfValueTF{#2}
      {\oldsection[#2]{#3}}% \section[.]{..}
      {\oldsection{#3}}% \section{..}
    \NR@gettitle{#3}% enable \nameref
  }%
}

\let\oldsubsection\subsection
\RenewDocumentCommand{\subsection}{s o m}{%
  \IfBooleanTF{#1}{%
    \phantomsection%
    \oldsubsection*{#3}% \subsection*{..}
    \NR@gettitle{#3}% enable \nameref
    \addcontentsline{toc}{subsection}{#3}% starred subsections are listed manually
  }{%
    \IfValueTF{#2}
      {\oldsubsection[#2]{#3}}% \subsection[.]{..}
      {\oldsubsection{#3}}% \subsection{..}
    \NR@gettitle{#3}% enable \nameref
  }%
}

\makeatother

% Use thesis-style hierarchical numbering for subsections.  If a
% subsection is used before any numbered section (as in the current
% appendix example), fall back to a clean top-level appendix-style label.
\renewcommand{\thesubsection}{%
  \ifnum\value{section}=0 \Alph{subsection}\else\thesection.\arabic{subsection}\fi}
\renewcommand{\thesubsubsection}{\thesubsection.\arabic{subsubsection}}
% Keep hyperref anchors clean and unique when appendices contain
% subsections before any numbered appendix section.
\providecommand{\theHsection}{\thesection}
\renewcommand{\theHsubsection}{%
  \ifnum\value{section}=0 appendix.\arabic{subsection}%
  \else\theHsection.\arabic{subsection}\fi}
\renewcommand{\theHsubsubsection}{\theHsubsection.\arabic{subsubsection}}

\titleformat{\section}
  {\large\sffamily\bfseries}
  {\thesection.}
  {0.5em}
  {#1}
  []
\titleformat{name=\section,numberless}
  {\large\sffamily\bfseries}
  {}
  {0em}
  {#1}
  []
\titleformat{\subsection}[runin]
  {\sffamily\bfseries}
  {\thesubsection.}
  {0.5em}
  {#1. }
  []
\titleformat{\subsubsection}[runin]
  {\sffamily\small\bfseries\itshape}
  {\thesubsubsection.}
  {0.5em}
  {#1. }
  []
\titleformat{\paragraph}[runin]
  {\sffamily\small\bfseries}
  {}
  {0em}
  {#1}
\titlespacing*{\section}{0pc}{3ex plus4pt minus3pt}{5pt}
\titlespacing*{\subsection}{0pc}{2.5ex plus3pt minus2pt}{2pt}
\titlespacing*{\subsubsection}{0pc}{2ex plus2.5pt minus1.5pt}{2pt}
\titlespacing*{\paragraph}{0pc}{1.5ex plus2pt minus1pt}{12pt}

%% Article meta data additional fields
\makeatletter
\newbool{@diploma}

\newcommand{\program}[1]{%
  \ifstrequal{#1}{BUN-RI}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Univerzitetni študijski program Računalništvo in informatika}
    \gdef\teza@program@longname@english{Academic study programme Computer and Information Science}
    \setbool{@diploma}{true}
    }{%
  \ifstrequal{#1}{BVS-RI}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Visokošolski strokovni študijski program Računalništvo in informatika}
    \gdef\teza@program@longname@english{Professional study programme Computer and Information Science}
    \setbool{@diploma}{true}
  }{%
  \ifstrequal{#1}{BUN-RM}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Univerzitetni interdisciplinarni študijski program Računalništvo in matematika}
    \gdef\teza@program@longname@english{Academic interdiciplinary study programme Computer Science and Mathematics}
    \setbool{@diploma}{true}
  }{%
  %\ifstrequal{#1}{BUN-MM}{\gdef\teza@program{#1}\gdef\teza@program@longname{Univerzitetni študijski program prve stopnje Multimedija}\setbool{@diploma}{true}}{%
  \ifstrequal{#1}{BMA-RI-PV}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Magistrski študijski program Računalništvo in informatika}
    \gdef\teza@program@longname@english{Master´s study programme Computer and Information Science}
    \gdef\teza@track{Podatkovne vede}
    \gdef\teza@track@english{Data Science}
    \setbool{@diploma}{false}
    }{%
  \ifstrequal{#1}{BMA-RI}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Magistrski študijski program Računalništvo in informatika}
    \gdef\teza@program@longname@english{Master´s study programme Computer and Information Science}
    \gdef\teza@track{Računalništvo in informatika}
    \gdef\teza@track@english{Computer and Information Science}
    \setbool{@diploma}{false}
    }{%
  \ifstrequal{#1}{BMA-MM}{
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Magistrski študijski program Multimedija}
    \gdef\teza@program@longname@english{Master´s study programme Multimedia}
    \setbool{@diploma}{false}
  }{%
  \ifstrequal{#1}{BMA-UI}{
    \@emaitrue
    \gdef\teza@program{#1}
    \gdef\teza@program@longname{Erasmus Mundus skupni magistrski študijski program Umetna inteligenca}
    \gdef\teza@program@longname@english{Erasmus Mundus Joint Master in Artificial Intelligence (EMAI)}
    \setbool{@diploma}{false}
    }{%
    \ClassError{friteza}{Invalid program '#1'}%
      {Allowed values are: BUN-RI, BVS-RI, BUN-RM, BMA-RI-PV, BMA-RI, BMA-MM, BMA-UI}%
  }}}}}}}
}

\newbool{@slovene}

\newcommand{\mainlanguage}[1]{%
  \ifstrequal{#1}{slovene}{
      \setbool{@slovene}{true}
  }{
    \ifstrequal{#1}{english}
    {
      \setbool{@slovene}{false}
    }
    {
    \ClassError{friteza}{Invalid language '#1'}%
      {Allowed values are: slovene, english}
    }
  }
}

\AtBeginDocument{
  \ifbool{@slovene}
    {
      \selectlanguage{slovene}
      \hypersetup{
          pdfinfo={
            Title={\@title@sl},
            Subject={zaključno delo},
            Author={\firstauthor},
            Keywords={\@keywords@sl}
          }
      }
    }
    {\selectlanguage{english}
      \hypersetup{
          pdfinfo={
            Title={\@title@en},
            Subject={thesis},
            Author={\firstauthor},
            Keywords={\@keywords@en}
          }
      }
    }
    \justifying
}

\newcommand{\sloabstract}[1]{\gdef\@abstract@sl{#1}}
\newcommand{\engabstract}[1]{\gdef\@abstract@en{#1}}

\AddToHook{env/document/before}{
    \setbool{singlecolumn}{true}
  \ifbool{@slovene}{
    \begin{abstract}
      \@abstract@sl
    \end{abstract}
    \significancestatement
        {\@title@en}
        {\@abstract@en}
        {\printkeywords[english]}
  }{
    \begin{abstract}
      \@abstract@en
    \end{abstract}
    \significancestatement
        {\@title@sl}
        {\@abstract@sl}
        {\printkeywords[slovene]}
  }
}

%\newcommand{\program}[1]{\def\teza@program{#1}}
\newcommand{\vol}[1]{\def\teza@vol{#1}}
\newcommand{\no}[1]{\def\teza@no{#1}}
\newcommand{\fraca}[1]{\def\@fraca{#1}\doi{frača: \href{http://fraca.si/\@fraca}{\color{black}\@fraca}}}
\newcommand{\additionalelement}[1]{\def\@additionalelement{#1}}
\newcommand{\dates}[3]{\def\date@day{#1}\def\date@month{#2}\def\date@year{#3}}
\newcommand{\doi}[1]{\def\@doi{#1}}
\newcommand{\leadauthor}[1]{\def\@leadauthor{#1}}
\newcommand{\etal}[1]{\def\@etal{#1}}

\ExplSyntaxOn

\ExplSyntaxOn

\clist_new:N \@keywords@sl
\clist_new:N \@keywords@en

\cs_new:Npn \friteza_keywords_separator:
  { \nobreakspace\textbar\nobreakspace }

\cs_new_protected:Npn \friteza_keywords_invalid_language:n #1
  {
    \ClassError { friteza }
      { Invalid~language~'#1' }
      { Allowed~values~are:~slovene,~english }
  }

\NewDocumentCommand \keywords { O{} m }
  {
    \str_case:nnF { #1 }
      {
        { slovene }
          { \clist_gset:Nn \@keywords@sl { #2 } }

        { english }
          { \clist_gset:Nn \@keywords@en { #2 } }
      }
      {
        \friteza_keywords_invalid_language:n { #1 }
      }
  }

\NewDocumentCommand \printkeywords { O{} }
  {
    \str_case:nnF { #1 }
      {
        { slovene }
          {
            \clist_if_empty:NF \@keywords@sl
              {
                \clist_use:Nn \@keywords@sl
                  { \friteza_keywords_separator: }
              }
          }

        { english }
          {
            \clist_if_empty:NF \@keywords@en
              {
                \clist_use:Nn \@keywords@en
                  { \friteza_keywords_separator: }
              }
          }
      }
      {
        \friteza_keywords_invalid_language:n { #1 }
      }
  }

\ExplSyntaxOff

\newcommand{\authorcontributions}[1]{\def\@authorcontributions{#1}}
\newcommand{\authordeclaration}[1]{\def\@authordeclaration{#1}}
\newcommand{\equalauthors}[1]{\def\@equalauthors{#1}}
\newcommand{\correspondingauthor}[1]{\def\@correspondingauthor{#1}}
\newcommand{\significancestatement}[3]{\def\@significancestatement{\footnotesize{\bfseries #1}\par\vspace{1mm}#2\par\vspace{1mm}#3\par}}
\newcommand{\matmethods}[2]{\def\mathmethods@name{#1}\def\@matmethods{#2}}
\newcommand{\acknow}[1]{\def\@acknow{#1}}

\newcommand{\supervisor}[2]{\def\supervisortitle{#1}\def\supervisorname{#2}}
\newcommand{\repository}[1]{\def\@repo{#1}}

\ExplSyntaxOn
\newcommand{\cosupervisor}[3][]{
    \def\cosupervisortitle{#2}
    \def\cosupervisorname{#3}
    \tl_if_blank:eTF {#1} {} {
        \def\cosupervisoraffiliation{#1}
    }
}
\ExplSyntaxOff

\def\firstauthor{}

\let\authblk@author\author
\renewcommand{\author}{\@ifnextchar[{\@author@opt}{\@author@noopt}}

\def\@author@opt[#1]#2{%
  \ifx\firstauthor\@empty
    \gdef\firstauthor{#2}%
  \fi
  \authblk@author[#1]{#2}%
}

\def\@author@noopt#1{%
  \ifx\firstauthor\@empty
    \gdef\firstauthor{#1}%
  \fi
  \authblk@author{#1}%
}

\makeatother

%% Dropped capital for first letter of main text
\newcommand{\dropcap}[1]{\lettrine[lines=2,lraise=0.05,findent=0.1em, nindent=0em]{{\dropcapfont{#1}}}{}}

%% Abstract formatting
\def\xabstract{abstract}
\long\def\abstract#1\end#2{\def\two{#2}\ifx\two\xabstract
\long\gdef\theabstract{\ignorespaces#1}
\def\go{\end{abstract}}\else
\typeout{^^J^^J PLEASE DO NOT USE ANY \string\begin\space \string\end^^J
COMMANDS WITHIN ABSTRACT^^J^^J}#1\end{#2}
\gdef\theabstract{\vskip12pt BADLY FORMED ABSTRACT: PLEASE DO
NOT USE {\tt\string\begin...\string\end} COMMANDS WITHIN
THE ABSTRACT\vskip12pt}\let\go\relax\fi
\go}

% Define an environment with abstract content and styling
\makeatletter
\newcommand{\abscontent}{
\noindent
{%
\parbox{\dimexpr\linewidth}{%
    \vskip3.5pt%
	\absfont \color{black80}\theabstract
}%
}%
\vskip10pt%
\noindent
{\parbox{\dimexpr\linewidth}{%
\ifbool{@slovene}
{
 \keywordsfont \color{P1797}\@ifundefined{@keywords@sl}{}{\printkeywords[slovene]}%
}{
   \keywordsfont \color{P1797}\@ifundefined{@keywords@en}{}{\printkeywords[english]}
}
}}%
\vskip12pt%
}
\makeatother

% Option to format abstract differently for certain layouts (not used)
\newcommand{\abscontentformatted}{
\abscontent
}

%% Manual adjustment to line up main content with line numbers
\makeatletter
\newlength\pnas@vertadjust
\newcommand\verticaladjustment[1]{\setlength{\pnas@vertadjust}{#1}}

%% Custom title page
\renewcommand{\@maketitle}{%
  \verticaladjustment{-2pt}
{
    {\raggedright\color{black80}\baselineskip= 24pt\titlefont {\expandafter\ifstrequal\expandafter{\languagename}{slovene}{\@title@sl}{\@title@en}} \par}%
    \vskip10pt% between base of title and base of author line
    {\raggedright \color{P1797}\sffamily\bfseries \firstauthor \par}
    \vskip6pt% between base of title and base of author line
    {\raggedright \footnotesize\sffamily {\expandafter\ifstrequal\expandafter{\languagename}{slovene}{\teza@program@longname}{\teza@program@longname@english}}{\@ifundefined{teza@track}{\par}{\raggedright \sffamily\footnotesize{, \expandafter\ifstrequal\expandafter{\languagename}{slovene}{smer \teza@track}{study field \teza@track@english}\par}}}}
\vskip12pt%
\noindent\begin{tikzpicture}[overlay, remember picture]
\node[anchor=north west] at (-2.25in,0pt) {%
	\begin{minipage}{2in}
     \sffamily\footnotesize%
      {\color{black80}\textbf{%
          {\expandafter\ifstrequal\expandafter{\languagename}{slovene}
          {Fakulteta za računalništvo in informatiko}
          {Faculty of Computer and Information Science}
      }}}\par
    {\expandafter\ifstrequal\expandafter{\teza@program}{BUN-RM}{\textbf{\color{black80}\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Fakulteta za matematiko in fiziko}{Faculty of Mathematics and Physics}}\par}{}}
	{\expandafter\ifstrequal\expandafter{\teza@program}{BMA-MM}{\textbf{\color{black80}\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Fakulteta za elektrotehniko}{Faculty of Electrical Engineering}}\par}{}}
	\vskip6pt
	{{{\color{black80}\textbf{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Mentor}{Supervisor}:}} \@ifundefined{supervisorname}{}{\supervisortitle\  \supervisorname}}}\par
	{\@ifundefined{cosupervisorname}{}{{{\color{black80}\textbf{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{Somentor}{Cosupervisor}:}} \cosupervisortitle\ \cosupervisorname \@ifundefined{cosupervisoraffiliation}{}{, \cosupervisoraffiliation}}
    }}\par
	\vskip24pt
	{{{\color{black80}\textbf{Copyright:}} \expandafter\ifstrequal\expandafter{\languagename}{slovene}{To delo je ponujeno pod licenco CC-BY-4.0.}{This work is licensed under a CC-BY-4.0 license.}\@ifundefined{@repo}{}{\expandafter\ifstrequal\expandafter{\languagename}{slovene}{ Razvita programska oprema in rezultati dela so na voljo v okviru GPL-3.0-or-later na}{ The software and results produced in this work are released under GPL-3.0-or-later at }
    {\@repo.}}}}\par
      \end{minipage}};
    \end{tikzpicture}
\begin{tikzpicture}[overlay, remember picture]
\path (current page.south west) ++(38.5pt,32pt+25pt+2pt) node[anchor=south west] {\begin{minipage}{2in}
      \ifbool{@slovene}{\selectlanguage{english}}{\selectlanguage{slovene}}
      {\sffamily \@ifundefined{@significancestatement}{}{
      \@significancestatement}
      \ifbool{@slovene}{\selectlanguage{slovene}}{\selectlanguage{english}}
    }
	\end{minipage}};
\node[anchor=north west] at ([shift={(-0pt,-32pt-1pt)}]current page.north west) {\includegraphics[scale=0.18]{logo/UL-FRI.png}};
\end{tikzpicture}
}%
\vskip\pnas@vertadjust
}%

\AddToHook{cmd/maketitle/before}{%
  \if@emai
    \InputIfFileExists{style/EMAI_title_page.tex}{}%
      {\ClassWarning{friteza}{EMAI program selected, but EMAI_title_page.tex was not found}}%
    \clearpage
    \setcounter{page}{1}% keep the existing thesis title page numbered as before
  \fi
}

\AddToHook{cmd/maketitle/after}{
  \thispagestyle{firststyle}

  \abscontentformatted
}

%% Footnotes set up
\RequirePackage[flushmargin,ragged]{footmisc}
\renewcommand*{\footnotelayout}{\normalfont\sffamily\fontsize{6}{8}\selectfont} % set the footnote font
\renewcommand{\footnoterule}{% Set the footnote hrule style
  \kern -3pt
  {\color{black50} \hrule width 72pt height 0.25pt}
  \kern 2.5pt
}

%% Set up the acknowledgments field
% Place this custom heading after the deepest standard level, so titlesec
% does not demote \subsection and \subsubsection below secnumdepth.
\titleclass{\acknow@section}{straight}[\subparagraph]
\newcounter{acknow@section}
\providecommand*{\toclevel@acknow@section}{1}

% Format is set for research articles by default
\titleformat{\acknow@section}[runin]
   {\sffamily\normalsize\bfseries}
   {}
   {0em}
   {#1.}
   []
\titlespacing{\acknow@section}
	{0pt}
	{3.25ex plus 1ex minus .2ex}
	{1.5ex plus .2ex}
\newcommand{\showacknow}{% Display acknowledgments section
  \@ifundefined{@acknow}{}{
    \expandafter\ifstrequal\expandafter{\languagename}{slovene}
      {\acknow@section{ZAHVALA}\small\@acknow}
      {\acknow@section{ACKNOWLEDGEMENT}\small\@acknow}
  }
}

%% Set up the materials&methods field
% Place this custom heading after the deepest standard level for the same
% reason as \acknow@section above: keep standard headings numberable.
\titleclass{\matmethods@section}{straight}[\subparagraph]
\newcounter{matmethods@section}
\providecommand*{\toclevel@matmethods@section}{0}
% Format is set for research articles by default
\titleformat{\matmethods@section}
   {\sffamily\normalsize\bfseries}
   {}
   {0em}
   {#1}
   []
\titlespacing{\matmethods@section}
	{0pt}
	{3.25ex plus 1ex minus .2ex}
	{1.5ex plus .2ex}
\newcommand{\showmatmethods}{% Display materials&methods section
\@ifundefined{@matmethods}{}{\matmethods@section{\mathmethods@name}{\small\noindent\@matmethods}}
}

%% Other packages
\RequirePackage{enumitem} % For reducing bullet list item separation

\newlist{compactitemize}{itemize}{3}
\setlist[compactitemize, 1]{label=\textbullet, noitemsep}
\setlist[compactitemize, 2]{label=\normalfont\bfseries \textendash, noitemsep}
\setlist[compactitemize, 3]{label=\textasteriskcentered, noitemsep}
\setlist[compactitemize, 4]{label=\textperiodcentered, noitemsep}

\newlist{compactenumerate}{enumerate}{3}
\setlist[compactenumerate]{label=\arabic*.,noitemsep}

\newenvironment{longsloveneabstract}{
    \selectlanguage{slovene}
    \section*{Daljši slovenski povzetek}
  }
  {
    \selectlanguage{english}
  }

% Setting up the appendix
\RequirePackage[page]{appendix}

\makeatletter
\renewcommand{\@sec@pppage}{%
    \par
    \addvspace{4ex}%
    \@afterindentfalse
    {\parindent \z@ \raggedright
    \interlinepenalty \@M
    \normalfont
    {\large\sffamily\bfseries \expandafter\ifstrequal\expandafter{\languagename}{slovene}{Priloge}{Appendices}} %
    \markboth{}{}\par}
    \ifbool{@slovene}{
      \addcontentsline{toc}{section}{Priloge}
    }{
      \addcontentsline{toc}{section}{Appendices}
    }
    \nobreak
    \vskip 3ex
    \@afterheading
    }
\makeatother

\RequirePackage{minted}
\RequirePackage{siunitx}

\newlength{\largefloatoffset}
\setlength{\largefloatoffset}{2.25in}

\newenvironment{largefigure}[1][tbp]{
  \begin{figure}[#1]
  \begin{adjustwidth}{-2.25in}{0in}
  \centering
}{
  \end{adjustwidth}
  \end{figure}
}

\newenvironment{largetable}[1][tbp]{%
  \begin{table}[#1]%
  \begin{adjustwidth}{-\largefloatoffset}{0pt}%
  \centering
}{%
  \end{adjustwidth}%
  \end{table}%
}

\AddToHook{cmd/bibliography/before}{
  \showacknow

  \begin{multicols}{2}
}

\AddToHook{cmd/bibliography/after}{
  \end{multicols}
}

\templatetype{style/pnasresearcharticle}


% theorem environments
\RequirePackage{amsthm}
\theoremstyle{definition}
\newtheorem{axiom}{\protect\axiomname}
\newtheorem{definition}{\protect\definitionname}
\newtheorem{remark}[definition]{\protect\remarkname}
\newtheorem{example*}[definition]{\protect\examplename}
\newenvironment{example}[1][]
	{\begin{example*}
	  [#1]\renewcommand*{\qedsymbol}{$\diamondsuit$}\pushQED{\qed}
	}{
	\popQED
	 \end{example*}}

\theoremstyle{plain}
\newtheorem{theorem}[definition]{\protect\theoremname}
\newtheorem{lemma}[definition]{\protect\lemmaname}
\newtheorem{proposition}[definition]{\protect\propositionname}
\newtheorem{corollary}[definition]{\protect\corollaryname}

\newcommand{\theoremname}{}
\newcommand{\exaname}{} % initialization
\newcommand{\lemmaname}{}
\newcommand{\axiomname}{}
\newcommand{\definitionname}{}
\newcommand{\remarkname}{}
\newcommand{\examplename}{}
\newcommand{\propositionname}{}
\newcommand{\corollaryname}{}
\makeatletter
\newcommand*{\friteza@setalgorithmcaption}[2]{%
  \@ifundefined{floatname}{}{\floatname{algorithm}{#1}}%
  \@ifundefined{listalgorithmname}{}{\renewcommand{\listalgorithmname}{#2}}%
}
\newcommand*{\friteza@setlistingcaption}[1]{%
  \@ifundefined{listingscaption}{}{\renewcommand{\listingscaption}{#1}}%
}
\addto\captionsenglish{%
  \renewcommand{\theoremname}{Theorem}%
  \renewcommand{\lemmaname}{Lemma}%
  \renewcommand{\axiomname}{Axiom}
  \renewcommand{\definitionname}{Definition}
  \renewcommand{\remarkname}{Remark}
  \renewcommand{\examplename}{Example}
  \renewcommand{\propositionname}{Proposition}
  \renewcommand{\corollaryname}{Corollary}
  \friteza@setalgorithmcaption{Algorithm}{List of Algorithms}%
  \friteza@setlistingcaption{Listing}%
}
\addto\captionsslovene{%
  \renewcommand{\theoremname}{Izrek}%
  \renewcommand{\lemmaname}{Lema}%
  \renewcommand{\axiomname}{Aksiom}
  \renewcommand{\definitionname}{Definicija}
  \renewcommand{\remarkname}{Opomba}
  \renewcommand{\examplename}{Zgled}
  \renewcommand{\propositionname}{Trditev}
  \renewcommand{\corollaryname}{Posledica}
  \friteza@setalgorithmcaption{Algoritem}{Seznam algoritmov}%
  \friteza@setlistingcaption{Izpis}%
}
\makeatother
