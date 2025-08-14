import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";

export async function POST(req: NextRequest) {
  const form = await req.formData();
  const file = form.get("file") as File;
  if (!file) return NextResponse.json({ error: "No file" }, { status: 400 });
  const buf = Buffer.from(await file.arrayBuffer());
  const sha = crypto.createHash("sha256").update(buf).digest("hex");

  // store to blob/S3 or temp folder; here we'll just echo back for demo
  // const url = await uploadToBlob(file)

  return NextResponse.json({ assetId: sha.slice(0,12), fingerprint: sha, size: buf.length });
}


