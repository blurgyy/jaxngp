# This file was generated by nvfetcher, please do not modify it manually.
{ fetchgit, fetchurl, fetchFromGitHub, dockerTools }:
{
  pyimgui = {
    pname = "pyimgui";
    version = "2.0.0";
    src = fetchurl {
      url = "https://pypi.io/packages/source/i/imgui/imgui-2.0.0.tar.gz";
      sha256 = "sha256-L7247tO429fqmK+eTBxlgrC8TalColjeFjM9jGU9Z+E=";
    };
  };
  tyro = {
    pname = "tyro";
    version = "0.4.2";
    src = fetchurl {
      url = "https://pypi.io/packages/source/t/tyro/tyro-0.4.2.tar.gz";
      sha256 = "sha256-8UaFs/mMsQKOOiuSpNrpvkRA0YmXDqTi7IIN3TV/Ti4=";
    };
  };
}
