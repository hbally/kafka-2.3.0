/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THIS CODE IS AUTOMATICALLY GENERATED.  DO NOT EDIT.

package org.apache.kafka.common.message;

import org.apache.kafka.common.errors.UnsupportedVersionException;
import org.apache.kafka.common.protocol.ApiMessage;
import org.apache.kafka.common.protocol.MessageUtil;
import org.apache.kafka.common.protocol.Readable;
import org.apache.kafka.common.protocol.Writable;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.protocol.types.Schema;
import org.apache.kafka.common.protocol.types.Struct;
import org.apache.kafka.common.protocol.types.Type;


public class HeartbeatRequestData implements ApiMessage {
    private String groupId;
    private int generationId;
    private String memberId;
    private String groupInstanceId;
    
    public static final Schema SCHEMA_0 =
        new Schema(
            new Field("group_id", Type.STRING, "The group id."),
            new Field("generation_id", Type.INT32, "The generation of the group."),
            new Field("member_id", Type.STRING, "The member ID.")
        );
    
    public static final Schema SCHEMA_1 = SCHEMA_0;
    
    public static final Schema SCHEMA_2 = SCHEMA_1;
    
    public static final Schema SCHEMA_3 =
        new Schema(
            new Field("group_id", Type.STRING, "The group id."),
            new Field("generation_id", Type.INT32, "The generation of the group."),
            new Field("member_id", Type.STRING, "The member ID."),
            new Field("group_instance_id", Type.NULLABLE_STRING, "The unique identifier of the consumer instance provided by end user.")
        );
    
    public static final Schema[] SCHEMAS = new Schema[] {
        SCHEMA_0,
        SCHEMA_1,
        SCHEMA_2,
        SCHEMA_3
    };
    
    public HeartbeatRequestData(Readable readable, short version) {
        read(readable, version);
    }
    
    public HeartbeatRequestData(Struct struct, short version) {
        fromStruct(struct, version);
    }
    
    public HeartbeatRequestData() {
        this.groupId = "";
        this.generationId = 0;
        this.memberId = "";
        this.groupInstanceId = null;
    }
    
    @Override
    public short apiKey() {
        return 12;
    }
    
    @Override
    public short lowestSupportedVersion() {
        return 0;
    }
    
    @Override
    public short highestSupportedVersion() {
        return 3;
    }
    
    @Override
    public void read(Readable readable, short version) {
        this.groupId = readable.readNullableString();
        this.generationId = readable.readInt();
        this.memberId = readable.readNullableString();
        if (version >= 3) {
            this.groupInstanceId = readable.readNullableString();
        } else {
            this.groupInstanceId = null;
        }
    }
    
    @Override
    public void write(Writable writable, short version) {
        writable.writeString(groupId);
        writable.writeInt(generationId);
        writable.writeString(memberId);
        if (version >= 3) {
            writable.writeNullableString(groupInstanceId);
        }
    }
    
    @Override
    public void fromStruct(Struct struct, short version) {
        this.groupId = struct.getString("group_id");
        this.generationId = struct.getInt("generation_id");
        this.memberId = struct.getString("member_id");
        if (version >= 3) {
            this.groupInstanceId = struct.getString("group_instance_id");
        } else {
            this.groupInstanceId = null;
        }
    }
    
    @Override
    public Struct toStruct(short version) {
        Struct struct = new Struct(SCHEMAS[version]);
        struct.set("group_id", this.groupId);
        struct.set("generation_id", this.generationId);
        struct.set("member_id", this.memberId);
        if (version >= 3) {
            struct.set("group_instance_id", this.groupInstanceId);
        }
        return struct;
    }
    
    @Override
    public int size(short version) {
        int size = 0;
        size += 2;
        size += MessageUtil.serializedUtf8Length(groupId);
        size += 4;
        size += 2;
        size += MessageUtil.serializedUtf8Length(memberId);
        if (version >= 3) {
            size += 2;
            if (groupInstanceId != null) {
                size += MessageUtil.serializedUtf8Length(groupInstanceId);
            }
        } else {
            if (groupInstanceId != null) {
                throw new UnsupportedVersionException("Attempted to write a non-default groupInstanceId at version " + version);
            }
        }
        return size;
    }
    
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof HeartbeatRequestData)) return false;
        HeartbeatRequestData other = (HeartbeatRequestData) obj;
        if (this.groupId == null) {
            if (other.groupId != null) return false;
        } else {
            if (!this.groupId.equals(other.groupId)) return false;
        }
        if (generationId != other.generationId) return false;
        if (this.memberId == null) {
            if (other.memberId != null) return false;
        } else {
            if (!this.memberId.equals(other.memberId)) return false;
        }
        if (this.groupInstanceId == null) {
            if (other.groupInstanceId != null) return false;
        } else {
            if (!this.groupInstanceId.equals(other.groupInstanceId)) return false;
        }
        return true;
    }
    
    @Override
    public int hashCode() {
        int hashCode = 0;
        hashCode = 31 * hashCode + (groupId == null ? 0 : groupId.hashCode());
        hashCode = 31 * hashCode + generationId;
        hashCode = 31 * hashCode + (memberId == null ? 0 : memberId.hashCode());
        hashCode = 31 * hashCode + (groupInstanceId == null ? 0 : groupInstanceId.hashCode());
        return hashCode;
    }
    
    @Override
    public String toString() {
        return "HeartbeatRequestData("
            + "groupId='" + groupId + "'"
            + ", generationId=" + generationId
            + ", memberId='" + memberId + "'"
            + ", groupInstanceId='" + groupInstanceId + "'"
            + ")";
    }
    
    public String groupId() {
        return this.groupId;
    }
    
    public int generationId() {
        return this.generationId;
    }
    
    public String memberId() {
        return this.memberId;
    }
    
    public String groupInstanceId() {
        return this.groupInstanceId;
    }
    
    public HeartbeatRequestData setGroupId(String v) {
        this.groupId = v;
        return this;
    }
    
    public HeartbeatRequestData setGenerationId(int v) {
        this.generationId = v;
        return this;
    }
    
    public HeartbeatRequestData setMemberId(String v) {
        this.memberId = v;
        return this;
    }
    
    public HeartbeatRequestData setGroupInstanceId(String v) {
        this.groupInstanceId = v;
        return this;
    }
}
